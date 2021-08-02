import os

from loader.dataset import Dataset
from loader.utils import load_kitti_odom_intrinsics


class KITTIOdom(Dataset):
    def __init__(self, img_dir, calib_dir, height, width, ext):
        super(KITTIOdom, self).__init__(img_dir, calib_dir, height, width, ext)

    def get_intrinsics_param(self):
        K = load_kitti_odom_intrinsics(os.path.join(self.calib_dir, "calib.txt"), self.height, self.width)[2]

        return K
