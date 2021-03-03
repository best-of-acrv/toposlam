import os
import cv2


class Dataset():
    def __init__(self, img_dir, calib_dir, height, width, ext):
        self.img_dir = img_dir
        self.calib_dir = calib_dir
        self.height = height
        self.width = width
        self.ext = ext

        self.img_id_list = []

    def get_id_list(self):
        """
        get all the image id in time order
        """
        img_names = sorted(os.listdir(self.img_dir))
        name_len = len(img_names[0])
        ext_len = len(self.ext)
        id_len = name_len - ext_len - 1
        for i in range(len(img_names)):
            img_name = img_names[i]
            self.img_id_list.append(img_name[:id_len])

        return self.img_id_list

    def get_image(self, img_id):
        img_path = os.path.join(self.img_dir, img_id) + '.' + self.ext
        img = cv2.imread(img_path)

        return img

    def get_intrinsics_param(self):
        """
        get the camera intrinsic parameters
        :return:
        """

        raise NotImplementedError



