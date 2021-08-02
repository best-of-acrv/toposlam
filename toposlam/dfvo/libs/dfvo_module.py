''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2019-01-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
LastEditTime: 2020-09-25
LastEditors: Huangying Zhan
@Description: DF-VO core program
'''

import cv2
import copy
from glob import glob
import math
from matplotlib import pyplot as plt
import numpy as np
import os
from time import time
from tqdm import tqdm

from dfvo.libs.geometry.camera_modules import SE3, Intrinsics
import dfvo.libs.datasets as Dataset
from dfvo.libs.deep_models.deep_models import DeepModel
from dfvo.libs.general.frame_drawer import FrameDrawer
from dfvo.libs.general.timer import Timer
from dfvo.libs.matching.keypoint_sampler import KeypointSampler
from dfvo.libs.matching.depth_consistency import DepthConsistency
from dfvo.libs.tracker import EssTracker, PnpTracker
from dfvo.libs.general.utils import *


class DFVO_Mod():
    def __init__(self, cfg, K):
        """
        Args:
            cfg (edict): configuration reading from yaml file
            K (list): intrinsics parameters, [cx, cy, fx, fy]
        """
        # configuration
        self.cfg = cfg

        # camera intrinsics
        self.cam_intrinsics = Intrinsics(K)

        self.setup()
        self.tracking_mode = "None"

    def setup(self):
        """Reading configuration and setup, including

            # - Timer
            # - Dataset
            - Tracking method
            - Keypoint Sampler
            - Deep networks
            - Deep layers
            - Visualizer
        """
        # get tracking method
        self.tracking_method = self.cfg.tracking_method
        self.initialize_tracker()

        # initialize keypoint sampler
        self.kp_sampler = KeypointSampler(self.cfg)

        # Deep networks
        self.deep_models = DeepModel(self.cfg)
        self.deep_models.initialize_models()

    def initialize_tracker(self):
        """Initialize tracker
        """
        if self.tracking_method == 'hybrid':
            self.e_tracker = EssTracker(self.cfg, self.cam_intrinsics)
            self.pnp_tracker = PnpTracker(self.cfg, self.cam_intrinsics)
        elif self.tracking_method == 'PnP':
            self.pnp_tracker = PnpTracker(self.cfg, self.cam_intrinsics)
        elif self.tracking_method == 'deep_pose':
            return
        else:
            assert False, "Wrong tracker is selected, choose from [hybrid, PnP, deep_pose]"

    def tracking(self):
        """Tracking using both Essential matrix and PnP
        Essential matrix for rotation and translation direction;
            *** triangluate depth v.s. CNN-depth for translation scale ***
        PnP if Essential matrix fails
        """

        ''' keypoint selection '''
        if self.tracking_method in ['hybrid', 'PnP']:
            # Depth consistency (CNN depths + CNN pose)
            if self.cfg.kp_selection.depth_consistency.enable:
                self.depth_consistency_computer.compute(self.cur_data, self.ref_data)

            # kp_selection
            kp_sel_outputs = self.kp_sampler.kp_selection(self.cur_data, self.ref_data)
            if kp_sel_outputs['good_kp_found']:
                self.kp_sampler.update_kp_data(self.cur_data, self.ref_data, kp_sel_outputs)

        ''' Pose estimation '''
        # Initialize hybrid pose
        hybrid_pose = SE3()
        E_pose = SE3()

        if not (kp_sel_outputs['good_kp_found']):
            # print("No enough good keypoints, constant motion will be used!")
            # pose = self.ref_data['motion']
            # self.update_global_pose(pose, 1)
            return False, None

        ''' E-tracker '''
        if self.tracking_method in ['hybrid']:
            # Essential matrix pose
            e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                self.ref_data[self.cfg.e_tracker.kp_src],
                self.cur_data[self.cfg.e_tracker.kp_src],
                not (self.cfg.e_tracker.iterative_kp.enable))  # pose: from cur->ref
            E_pose = e_tracker_outputs['pose']

            # Rotation
            hybrid_pose.R = E_pose.R

            # save inliers
            self.ref_data['inliers'] = e_tracker_outputs['inliers']

            # scale recovery
            if np.linalg.norm(E_pose.t) != 0:
                scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, False)
                scale = scale_out['scale']
                if self.cfg.scale_recovery.kp_src == 'kp_depth':
                    self.cur_data['kp_depth'] = scale_out['cur_kp_depth']
                    self.ref_data['kp_depth'] = scale_out['ref_kp_depth']
                    self.cur_data['rigid_flow_mask'] = scale_out['rigid_flow_mask']
                if scale != -1:
                    hybrid_pose.t = E_pose.t * scale

            # Iterative keypoint refinement
            if np.linalg.norm(E_pose.t) != 0 and self.cfg.e_tracker.iterative_kp.enable:
                # Compute refined keypoint
                self.e_tracker.compute_rigid_flow_kp(self.cur_data,
                                                     self.ref_data,
                                                     hybrid_pose)

                e_tracker_outputs = self.e_tracker.compute_pose_2d2d(
                    self.ref_data[self.cfg.e_tracker.iterative_kp.kp_src],
                    self.cur_data[self.cfg.e_tracker.iterative_kp.kp_src],
                    True)  # pose: from cur->ref
                E_pose = e_tracker_outputs['pose']

                # Rotation
                hybrid_pose.R = E_pose.R

                # save inliers
                self.ref_data['inliers'] = e_tracker_outputs['inliers']

                # scale recovery
                if np.linalg.norm(E_pose.t) != 0 and self.cfg.scale_recovery.iterative_kp.enable:
                    scale_out = self.e_tracker.scale_recovery(self.cur_data, self.ref_data, E_pose, True)
                    scale = scale_out['scale']
                    if scale != -1:
                        hybrid_pose.t = E_pose.t * scale
                else:
                    hybrid_pose.t = E_pose.t * scale
            self.tracking_mode = "Ess.Mat."

        ''' PnP-tracker '''
        if self.tracking_method in ['PnP', 'hybrid']:
            # PnP if Essential matrix fail
            if np.linalg.norm(E_pose.t) == 0 or scale == -1:
                pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                    self.ref_data[self.cfg.pnp_tracker.kp_src],
                    self.cur_data[self.cfg.pnp_tracker.kp_src],
                    self.ref_data['depth'],
                    not (self.cfg.pnp_tracker.iterative_kp.enable)
                )  # pose: from cur->ref

                # Iterative keypoint refinement
                if self.cfg.pnp_tracker.iterative_kp.enable:
                    self.pnp_tracker.compute_rigid_flow_kp(self.cur_data, self.ref_data, pnp_outputs['pose'])
                    pnp_outputs = self.pnp_tracker.compute_pose_3d2d(
                        self.ref_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                        self.cur_data[self.cfg.pnp_tracker.iterative_kp.kp_src],
                        self.ref_data['depth'],
                        True
                    )  # pose: from cur->ref

                # use PnP pose instead of E-pose
                hybrid_pose = pnp_outputs['pose']
                self.tracking_mode = "PnP"

        ''' Deep-tracker '''
        if self.tracking_method in ['deep_pose']:
            hybrid_pose = SE3(self.ref_data['deep_pose'])
            self.tracking_mode = "DeepPose"

        return True, hybrid_pose

    def deep_model_inference(self):
        """deep model prediction
        """
        if self.tracking_method in ['hybrid', 'PnP']:
            # Single-view Depth prediction
            # reference image
            img_list = [self.ref_data['img']]

            self.ref_data['raw_depth'] = \
                self.deep_models.forward_depth(imgs=img_list)
            self.ref_data['raw_depth'] = cv2.resize(self.ref_data['raw_depth'],
                                                    (self.cfg.image.width, self.cfg.image.height),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )
            self.ref_data['depth'] = preprocess_depth(self.ref_data['raw_depth'], self.cfg.crop.depth_crop,
                                                      [self.cfg.depth.min_depth, self.cfg.depth.max_depth])

            # current image
            img_list = [self.cur_data['img']]

            self.cur_data['raw_depth'] = \
                self.deep_models.forward_depth(imgs=img_list)
            self.cur_data['raw_depth'] = cv2.resize(self.cur_data['raw_depth'],
                                                    (self.cfg.image.width, self.cfg.image.height),
                                                    interpolation=cv2.INTER_NEAREST
                                                    )
            self.cur_data['depth'] = preprocess_depth(self.cur_data['raw_depth'], self.cfg.crop.depth_crop,
                                                      [self.cfg.depth.min_depth, self.cfg.depth.max_depth])

            # Two-view flow
            flows = self.deep_models.forward_flow(
                self.cur_data,
                self.ref_data,
                forward_backward=self.cfg.deep_flow.forward_backward)

            # Store flow
            self.ref_data['flow'] = flows[(self.ref_data['id'], self.cur_data['id'])].copy()
            if self.cfg.deep_flow.forward_backward:
                self.cur_data['flow'] = flows[(self.cur_data['id'], self.ref_data['id'])].copy()
                self.ref_data['flow_diff'] = flows[(self.ref_data['id'], self.cur_data['id'], "diff")].copy()

            return self.cur_data['raw_depth'], self.cur_data['flow']

    def main(self, img1, img2):
        """Main program

        Args:
            img1 (array, [HxWx3]): reference image
            img2 (array, [HxWx3]): current image

        Returns:
            valid (bool): True if enough keypoints are provided for solving pose
            pose (SE3): relatvie pose from view-2 to view-1; None if valid is False
        """
        ''' initialize data '''
        self.ref_data = {}
        self.cur_data = {}
        self.ref_data['img'] = img1
        self.cur_data['img'] = img2
        self.ref_data['id'] = 0
        self.cur_data['id'] = 1

        """ Deep model inferences """
        depth, flow = self.deep_model_inference()

        """ Visual odometry """
        valid, pose = self.tracking()

        return valid, pose, depth, flow


