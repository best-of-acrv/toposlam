from .dfvo.libs.dfvo_module import DFVO_Mod
from .dfvo.libs.geometry.camera_modules import SE3
from .dfvo.libs.geometry.pose_graph_optimizer import PoseGraphOptimizer
from .lcd.loop_closure_detect import LoopClosureDetect
from .loaders.kitti import KITTIOdom as Loader
from .vis.frame_drawer import FrameDrawer


class TopoSlam(object):

    def __init__(self, img_dir, calib_dir, cfg, device):
        # parameters
        self.img_dir = img_dir  # folder path of the image sequences for SLAM
        self.calib_dir = calib_dir  # folder path of the calibration file
        self.cfg = cfg  # cfg file
        self.device = device
        self.ext = self.cfg.image.ext  # image format extension
        self.height = self.cfg.image.height
        self.width = self.cfg.image.width

        self.tracking_stage = 0

        self.global_poses_pred = {0: SE3()}
        self.global_poses_opt = {0: SE3()}

        self.cur_data = {}
        self.ref_data = {}
        self.mat_data = {}
        self.lc_pairs = []

        # setup image loader module and camera intrinsic parameters
        self.loader = Loader(self.img_dir, self.calib_dir, self.height,
                             self.width, self.ext)
        self.img_id_list = self.loader.get_id_list()
        self.K = self.loader.get_intrinsics_param(
        )  # camera intrinsic: [cx, cy, fx, fy]

        # setup DFVO module
        self.dfvo = DFVO_Mod(self.cfg, self.K)

        # setup loop closure detection module
        self.loop_closure_detect = LoopClosureDetect(
            self.device, self.cfg.net_vlad.pretrained_model)
        # self.loop_closure_detect = LoopClosureDetect(device=device, net_vlad_ckp, belief_thr=0.15, candidate_num=5,
        #                                              frame_step_thr=200, belief_init_step=10)

        # setup pose graph optimization module
        self.pose_graph_optimize = PoseGraphOptimizer()

        # visualization interface
        self.drawer = FrameDrawer(self.K)

    def update_global_pose(self, new_pose, img_id, scale=1.):
        """update estimated poses w.r.t global coordinate system

        Args:
            new_pose (SE3): new pose
            scale (float): scaling factor
        """
        self.cur_data['pose'].t = self.cur_data[
            'pose'].R @ new_pose.t * scale + self.cur_data['pose'].t
        self.cur_data['pose'].R = self.cur_data['pose'].R @ new_pose.R
        self.global_poses_pred[img_id] = copy.deepcopy(self.cur_data['pose'])

    def update_data(self, ref_data, cur_data):
        """Update data

        Args:
            ref_data (dict): reference data
            cur_data (dict): current data

        Returns:
            ref_data (dict): updated reference data
            cur_data (dict): updated current data
        """
        for key in cur_data:
            if key == "id":
                ref_data['id'] = cur_data['id']
            else:
                if ref_data.get(key, -1) is -1:
                    ref_data[key] = {}
                ref_data[key] = cur_data[key]

        return ref_data, cur_data

    def main(self):
        """ main program """
        # start_frame = int(input("Start with frame: "))
        start_frame = 0
        old_pose = SE3()
        """ perform VO and pose graph construction """
        for img_id in tqdm(
                range(start_frame, len(self.img_id_list),
                      self.cfg.frame_step)):

            if self.tracking_stage == 0:  # the starting frame
                self.cur_data['id'] = self.img_id_list[img_id]
                self.cur_data['img'] = self.loader.get_image(
                    self.cur_data['id'])
                self.cur_data['pose'] = SE3()

                # pose graph
                self.pose_graph_optimize.add_vertex(
                    img_id,
                    g2o.Isometry3d(self.cur_data['pose'].R,
                                   self.cur_data['pose'].t), True)

                # loop closure detection, adding the current frame to the database
                _ = self.loop_closure_detect(
                    [self.cur_data['img'].transpose(2, 0, 1)], [img_id], False)

            elif self.tracking_stage >= 1:  # the second to the last frame
                self.cur_data['id'] = self.img_id_list[img_id]
                self.cur_data['img'] = self.loader.get_image(
                    self.cur_data['id'])

                # estimate relative pose
                valid, new_pose, depth, flow = self.dfvo.main(
                    self.ref_data['img'], self.cur_data['img'])
                if valid:
                    old_pose = copy.deepcopy(new_pose)
                else:
                    new_pose = old_pose

                self.cur_data['depth'] = depth
                self.cur_data['flow'] = np.transpose(flow, [1, 2, 0])

                # predict the pose of current frame w.r.t the global coordinate system
                self.update_global_pose(new_pose, img_id)

                # set the optimized pose as the predicted one, if loop closure is detected, all the optimized
                # poses will be updated
                self.global_poses_opt[img_id] = copy.deepcopy(
                    self.global_poses_pred[img_id])

                # pose graph
                self.pose_graph_optimize.add_vertex(
                    img_id,
                    g2o.Isometry3d(self.cur_data['pose'].R,
                                   self.cur_data['pose'].t), False)

                self.pose_graph_optimize.add_edge([img_id - 1, img_id],
                                                  g2o.Isometry3d(
                                                      new_pose.R, new_pose.t))

                # loop closure detection
                lcd_result = self.loop_closure_detect(
                    [self.cur_data['img'].transpose(2, 0, 1)], [img_id], True)

                # add new edge to the pose graph if loop closure is detected
                if lcd_result[0][0] is not None:
                    print(
                        'loop closure detection: matched pair of image {} and {} '
                        'with the belief of {:06f}\n'.format(
                            lcd_result[0][1], lcd_result[0][0],
                            lcd_result[0][2]))

                    self.mat_data['id'] = self.img_id_list[int(
                        lcd_result[0][1])]
                    # self.mat_data['query_id'] = copy.copy(self.cur_data['id'])
                    self.mat_data['img'] = self.loader.get_image(
                        self.mat_data['id'])

                    self.lc_pairs.append(
                        [int(lcd_result[0][1]),
                         int(lcd_result[0][0])])
                    valid, lcd_pose, _, _ = self.dfvo.main(
                        self.cur_data['img'], self.mat_data['img'])

                    if valid:
                        # adding edge to the pose graph
                        self.pose_graph_optimize.add_edge(
                            [img_id, int(lcd_result[0][1])],
                            g2o.Isometry3d(lcd_pose.R, lcd_pose.t))

                        # optimize the pose graph
                        self.pose_graph_optimize.optimize()

                        # update the optimized global poses
                        for img_id_opt in range(
                                start_frame + self.cfg.frame_step, img_id + 1,
                                self.cfg.frame_step):
                            self.global_poses_opt[img_id_opt] = SE3(
                                self.pose_graph_optimize.get_pose(img_id_opt))
                    else:
                        continue

                # visualization
                self.drawer.main(self.cur_data['img'], self.cur_data['depth'],
                                 self.cur_data['flow'], self.global_poses_pred,
                                 self.global_poses_opt, self.lc_pairs)

            # update reference and current data
            self.ref_data, self.cur_data = self.update_data(
                self.ref_data, self.cur_data)

            self.tracking_stage += 1

            # terminate the process from the interface
            if self.drawer.should_quit:
                print("Process terminated by the user!\n")
                break
        """ save trajectory"""
        result_dir = self.cfg.directory.result_dir
        if not os.path.isdir("{}".format(result_dir)):
            os.makedirs("{}")

        traj_txt_pred = "{}/poses_pred.txt".format(result_dir)
        global_poses_arr_pred = convert_SE3_to_arr(self.global_poses_pred)
        save_traj(traj_txt_pred, global_poses_arr_pred, format='kitti')
        print('save predicted poses to file {}\n'.format(traj_txt_pred))

        traj_txt_opt = "{}/poses_opt.txt".format(result_dir)
        global_poses_arr_opt = convert_SE3_to_arr(self.global_poses_opt)
        save_traj(traj_txt_opt, global_poses_arr_opt, format='kitti')
        print('save optimized poses to file {}\n'.format(traj_txt_opt))
