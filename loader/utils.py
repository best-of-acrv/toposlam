
def load_kitti_odom_intrinsics(file_name, new_h, new_w):
    """Load kitti odometry data intrinscis

    Args:
        file_name (str): txt file path

    Returns:
        intrinsics (dict): each element contains [cx, cy, fx, fy]
    """
    raw_img_h = 370.0
    raw_img_w = 1226.0
    intrinsics = {}
    with open(file_name, 'r') as f:
        s = f.readlines()
        for cnt, line in enumerate(s):
            line_split = [float(i) for i in line.split(" ")[1:]]
            intrinsics[cnt] = [
                line_split[2] / raw_img_w * new_w,
                line_split[6] / raw_img_h * new_h,
                line_split[0] / raw_img_w * new_w,
                line_split[5] / raw_img_h * new_h,
            ]
    return intrinsics
