import numpy as np
import OpenGL.GL as gl
import pangolin
import cv2
import matplotlib as mpl

from .utils import waitKey

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


class FrameDrawer():
    def __init__(self, K):

        # camera intrinsic
        self.K = K
        self.img_h = 200
        self.img_w = 700
        self.should_quit = False

        # Create display window
        pangolin.CreateWindowAndBind('DF-VO', 1280, 640)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1280, 640, 520, 520, 720, 250, 0.1, 5000),
            pangolin.ModelViewLookAt(0.1, -0.5, -2, 0, 0, 1, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        #  different model view matrix
        self.bird_view_matrix = pangolin.OpenGlMatrix()
        self.bird_view_matrix.m = np.array(
            [[9.87267818e-01, -3.53031899e-02, -1.55099774e-01, -1.84085547e+01],
             [7.72393359e-02, -7.45994709e-01, 6.61457466e-01, -8.69668877e+01],
             [-1.39055169e-01, -6.65015472e-01, -7.33769774e-01, -6.10378254e+02],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.drive_view_matrix = pangolin.OpenGlMatrix()
        self.drive_view_matrix.m = self.scam.GetModelViewMatrix().m

        # buttons for interactive visualization
        self.panel = pangolin.CreatePanel('ui')
        self.panel.SetBounds(0.78, 1.0, 0.0, 175 / 1280.)
        self.pause_ckbox = pangolin.VarBool('ui.pause(p/c)', value=False, toggle=True)
        self.bird_view_ckbox = pangolin.VarBool('ui.bird_view(b)', value=True, toggle=True)
        self.drive_view_ckbox = pangolin.VarBool('ui.drive_view(d)', value=False, toggle=True)
        self.opt_poses_ckbox = pangolin.VarBool('ui.opt_poses(o)', value=True, toggle=True)
        self.warped_view_ckbox = pangolin.VarBool('ui.warped_view(w)', value=True, toggle=True)

        # keyboard interface
        self.pause = False
        self.bird_view = True
        self.drive_view = False
        self.opt_poses = True
        self.warped_view = True

        # trajectory display handler
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 0.66, True)
        self.dcam.SetHandler(self.handler)

        # image display
        self.dimg = pangolin.Display('image')
        self.dimg.SetBounds(0.66, 0.99, 0.67, 1.0, 1280/640)
        self.dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        # depth display
        self.ddepth = pangolin.Display('depth')
        self.ddepth.SetBounds(0.33, 0.66, 0.67, 1.0, 1280 / 640)
        self.ddepth.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        # optical flow display
        self.dflow = pangolin.Display('flow')
        self.dflow.SetBounds(0.0, 0.33, 0.67, 1.0, 1280 / 640)
        self.dflow.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        # to upload display image
        self.texture = pangolin.GlTexture(self.img_w, self.img_h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    def rgbd_to_colored_pc(self, depth, rgb, f_x, f_y, c_x, c_y, cap=100):
        """Convert a pair of rgb and depth map to colored point cloud

        Args:
            depth (HxW np.ndarray): depth map
            rgb (HxWx3 np.ndarray): rgb image
            fx, fy, cx, cy (float, float, float, float): intrinsic params.
            cap (float): cap value for depth map
        Returns:
            points (Nx3 np.ndarray): points' position
            colors (Nx3 np.ndarray): color of each point
        """

        rgb_height, rgb_width, _ = rgb.shape
        x_map, y_map = np.meshgrid(np.arange(rgb_width), np.arange(rgb_height))
        xyz_rgb = np.concatenate(
            [x_map[:, :, None], y_map[:, :, None], depth[:, :, None], rgb],
            axis=2
        )
        xyz_rgb[:, :, 0] = (xyz_rgb[:, :, 0] - c_x) * xyz_rgb[:, :, 2] / f_x
        xyz_rgb[:, :, 1] = (xyz_rgb[:, :, 1] - c_y) * xyz_rgb[:, :, 2] / f_y
        points = xyz_rgb[:, :, :3].reshape(-1, 3)
        colors = xyz_rgb[:, :, 3:].reshape(-1, 3) / 255.
        cap_ind = np.logical_and(points[:, 2] < cap, points[:, 2] > 1)
        points = points[cap_ind]
        colors = colors[cap_ind]
        return points, colors

    def flow_to_image(self, flow, maxrad=-1):
        """Convert flow into middlebury color code image

        Args:
            flow (array, [HxWx2]): optical flow map

        Returns:
            img (array, [HxWx3]): optical flow image in middlebury color
        """
        u = flow[:, :, 0]
        v = flow[:, :, 1]

        idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
        u[idxUnknow] = 0
        v[idxUnknow] = 0

        if maxrad == -1:
            rad = np.sqrt(u ** 2 + v ** 2)
            maxrad = max(-1, np.max(rad))

        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)

        img = self.compute_color(u, v)

        idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)

    def compute_color(self, u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = self.make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u ** 2 + v ** 2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    def make_color_wheel(self):
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col + YG, 0] = 255 - np.transpose(
            np.floor(255 * np.arange(0, YG) / YG))
        colorwheel[col:col + YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC, 2] = np.transpose(
            np.floor(255 * np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col + CB, 1] = 255 - np.transpose(
            np.floor(255 * np.arange(0, CB) / CB))
        colorwheel[col:col + CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM, 0] = np.transpose(
            np.floor(255 * np.arange(0, BM) / BM))
        col += +BM

        # MR
        colorwheel[col:col + MR, 2] = 255 - np.transpose(
            np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col + MR, 0] = 255

        return colorwheel

    def depth_to_img(self, depth):
        """
        convert an depth map to a pseudo colored image for visulization
        """
        vis_depth = 1 / (depth + 1e-3)
        vmax = np.percentile(vis_depth, 90)
        normalizer = mpl.colors.Normalize(vmin=0, vmax=vmax)
        mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_img = (mapper.to_rgba(vis_depth)[:, :, :3] * 255).astype(np.uint8)

        return colormapped_img

    def convert_rgb_for_pango(self, rgb):
        """convert image as numpy array for pango visualization
        Args:
            rgb (HxWx3 np.ndarray): rgb image
        Return:
            rgb_pango (HxWx3 np.ndarray): transformed rgb image for pango
        """
        rgb_pango = rgb[::-1, :, :]  # horizontal flip for pango display
        rgb_pango = np.asarray(rgb_pango, order='C')
        return rgb_pango

    def draw(self, rgb, depth, flow, point_cloud, T_c_pre_pred, T_c_pre_opt, lc_xyz):
        """

        :param rgb:
        :param depth:
        :param flow:
        :param point_cloud:
        :param T_c_pre_pred:
        :param T_c_pre_opt:
        :param lc_xyz:
        :return:
        """

        # openGL initialize
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.95, 0.95, 0.95, 1.0)

        self.dcam.Activate(self.scam)

        if self.bird_view:
            self.scam.SetModelViewMatrix(self.bird_view_matrix)
        if self.drive_view:
            self.scam.SetModelViewMatrix(self.drive_view_matrix)

        # draw point cloud
        if self.warped_view:
            pangolin.DrawPoints(point_cloud[0], point_cloud[1])

        # draw predicted trajectory
        pose_num = len(T_c_pre_pred)
        gl.glLineWidth(4)
        gl.glColor3f(0.0, 1.0, 0.0)
        for i in range(pose_num):
            pangolin.DrawCamera(T_c_pre_pred[i], 0.2, 0.75, 0.8)

        #  draw optimized poses if selected
        if self.opt_poses:
            gl.glLineWidth(4)
            gl.glColor3f(0.0, 0.0, 1.0)
            for i in range(pose_num):
                pangolin.DrawCamera(T_c_pre_opt[i], 0.25, 0.75, 0.8)

        # draw lines between loop closure pairs
        if len(lc_xyz[0]) != 0:
            gl.glLineWidth(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawLines(lc_xyz[0], lc_xyz[1], point_size=5)

        # image display
        self.texture.Upload(self.convert_rgb_for_pango(rgb), gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.dimg.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        self.texture.RenderToViewport()

        # depth display
        self.texture.Upload(self.convert_rgb_for_pango(depth), gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.ddepth.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        self.texture.RenderToViewport()

        # flow display
        self.texture.Upload(self.convert_rgb_for_pango(flow), gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.dflow.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        self.texture.RenderToViewport()
        
        pangolin.FinishFrame()

    def interface(self):
        """
        keyboard interface
        """
        key = waitKey(10)

        if key == "p":  # pause the process
            self.pause = True
            self.pause_ckbox.SetVal(True)

        if key == "c":  # continue the process
            self.pause = False
            self.pause_ckbox.SetVal(False)

        if key == "b":  # bird-view mode
            self.bird_view = True
            self.drive_view = False
            self.bird_view_ckbox.SetVal(True)
            self.drive_view_ckbox.SetVal(False)

        if key == "d":  # drive-view mode
            self.bird_view = False
            self.drive_view = True
            self.bird_view_ckbox.SetVal(False)
            self.drive_view_ckbox.SetVal(True)

        if key == "o":  # switch for visualization of optimized poses
            self.opt_poses = not self.opt_poses
            self.opt_poses_ckbox.SetVal(self.opt_poses)

        if key == "w":  # switch for the warped view
            self.warped_view = not self.warped_view
            self.warped_view_ckbox.SetVal(self.warped_view)

    def main(self, rgb, depth, flow, poses_pred, poses_opt, lc_pairs):
        """
        the main interface of the frame drawer
        :param rgb:
        :param depth:
        :param flow:
        :param poses_pred:
        :param poses_opt:
        :param lc_pairs:
        :return:
        """

        rgb = rgb[:, :, ::-1]  # bgr->rgb

        # get the point cloud of the current frame
        points, colors = self.rgbd_to_colored_pc(depth, rgb, self.K[2], self.K[3], self.K[0], self.K[1])
        point_cloud = [points, colors]

        # covert depth and optical flow into pseud colored images and concatenate with the color image into one image
        colormapped_depth = cv2.resize(self.depth_to_img(depth), (self.img_w, self.img_h))
        colormapped_flow = cv2.resize(self.flow_to_image(flow), (self.img_w, self.img_h))
        rgb = cv2.resize(rgb, (self.img_w, self.img_h))

        # get the loop closure pairs
        xyz_1 = []
        xyz_2 = []
        if len(lc_pairs) != 0:
            cur_pose = poses_pred[len(poses_pred) - 1].pose
            for i in range(0, len(lc_pairs)):
                pose_1 = np.linalg.inv(cur_pose) @ np.asarray(poses_pred[lc_pairs[i][0]].pose)
                pose_2 = np.linalg.inv(cur_pose) @ np.asarray(poses_pred[lc_pairs[i][1]].pose)
                xyz_1.append(pose_1[:3, 3])
                xyz_2.append(pose_2[:3, 3])

        lc_xyz = [np.array(xyz_1), np.array(xyz_2)]

        # get the relative camera poses for visualization
        T_c_pre_pred = []
        T_c_pre_opt = []
        pose_num = len(poses_pred)

        cur_pose = poses_pred[pose_num - 1].pose
        for i in range(pose_num):
            pre_pose = poses_pred[i].pose
            T_c_pre_pred.append(np.linalg.inv(cur_pose) @ pre_pose)

        cur_pose = poses_opt[pose_num - 1].pose
        for i in range(pose_num):
            pre_pose = poses_opt[i].pose
            T_c_pre_opt.append(np.linalg.inv(cur_pose) @ pre_pose)

        # check the keyboard interface
        self.interface()

        # check if the pose is paused
        if self.pause:
            while True:  # iterative draw all the items until button "c" is pressed
                # terminate the process from the visualization interface
                if pangolin.ShouldQuit(): 
                    self.should_quit = True

                self.draw(rgb, colormapped_depth, colormapped_flow, point_cloud, T_c_pre_pred, T_c_pre_opt, lc_xyz)
                self.interface()
                if not self.pause:
                    break
        else:  # else only draw all the items one time
            # terminate the process from the visualization interface
            if pangolin.ShouldQuit(): 
                self.should_quit = True
            
            self.draw(rgb, colormapped_depth, colormapped_flow, point_cloud, T_c_pre_pred, T_c_pre_opt, lc_xyz)

        

