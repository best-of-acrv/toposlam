import argparse
import torch
import os

from slam import SLAM
from dfvo.libs.general.configuration import ConfigLoader


config_loader = ConfigLoader()


def read_cfgs():
    """Parse arguments and laod configurations

    Returns
    -------
    args : args
        arguments
    cfg : edict
        configuration dictionary
    """
    ''' Argument Parsing '''
    parser = argparse.ArgumentParser(description='SLAM system')
    parser.add_argument("-d", "--default_configuration", type=str,
                        default="cfg/default.yml",
                        help="default configuration files")
    parser.add_argument("-c", "--configuration", type=str,
                        default=None,
                        help="custom configuration file")
    parser.add_argument("-r", "--data_root", type=str, default="./data",
                        help="path containing image sequence directory")
    parser.add_argument("-s", "--seq", type=str, default="09",
                        help="which (kitti) image sequence to perform VO on")
    parser.add_argument("-e", "--ext", type=str, default="png",
                        help="file extension of the images")

    args = parser.parse_args()

    ''' Read configuration '''
    # read default and custom config, merge cfgs
    config_files = [args.default_configuration, args.configuration]
    cfg = config_loader.merge_cfg(config_files)

    return args, cfg


if __name__ == '__main__':
    # Read config
    args, cfg = read_cfgs()

    # use images in the "image_2" folder for VO
    img_dir = os.path.join(args.data_root, args.seq, 'image_2')

    # folder path that contains the calibration file
    calib_dir = os.path.join(args.data_root, args.seq)

    # use CUDA
    device = torch.device("cuda")

    """ perform VO """
    slam = SLAM(img_dir, calib_dir, cfg, device)
    slam.main()






