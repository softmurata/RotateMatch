import open3d as o3d
import numpy as np
import copy

# find loop => pcd match
# do not work well

# camera configuration
class Config(object):

    def __init__(self):
        self.width = 640
        self.height = 480
        self.fx = 381.694
        self.fy = 381.694
        self.cx = 323.56
        self.cy = 237.11
        self.bx = 0
        self.by = 0
        
        
def create_rotation_matrix(axis_angles):
    # angle
    anglex = axis_angles[0]
    angley = axis_angles[1]
    anglez = axis_angles[2]

    # x axis rotation
    anglex = anglex * np.pi / 180.0
    Rx = np.array([[1, 0, 0],
                [0, np.cos(anglex), -np.sin(anglex)],
                [0, np.sin(anglex), np.cos(anglex)]])

    # y axis rotation
    angley = angley * np.pi / 180.0
    Ry = np.array([[np.cos(angley), 0, np.sin(angley)],
                [0, 1, 0],
                [-np.sin(angley), 0, np.cos(angley)]])

    # z axis rotation
    anglez = anglez * np.pi / 180
    Rz = np.array([[np.cos(anglez), -np.sin(anglez), 0],
                [np.sin(anglez), np.cos(anglez), 0],
                [0,              0,             1]])

    R = Rz.dot(Ry).dot(Rx)
    
    print('Rz:', Rz)
    
    return R

def change_rotation_mat_for_open3d(R):
    R_s = np.zeros((4, 4))
    R_s[:3, :3] = R
    R_s[3, 3] = 1.0
    
    return R_s
    
        


dataset_dir = '../rgbd_dataset/RGBD/'
dataset_name = 'Nami7'
dataset_dir = dataset_dir + '{}/'.format(dataset_name)
rgb_dir = dataset_dir + 'rgb/'
depth_dir = dataset_dir + 'depth/'


start = 4
finish = 574

diff_angle = 360 / (finish - start)
angle = 0

angles = []

for i in range(start, finish):
    angles.append([i, angle])
    angle += diff_angle


target_num, anglez = angles[4]
axis_angles = [0, 0, anglez]
R = create_rotation_matrix(axis_angles)
Rs = change_rotation_mat_for_open3d(R)

print('R:', R)
print('Rs:', Rs)

source_rgb = o3d.io.read_image(rgb_dir + '{}.png'.format(start))
source_depth = o3d.io.read_image(depth_dir + '{}.png'.format(start))
# target
target_rgb = o3d.io.read_image(rgb_dir + '{}.png'.format(target_num))
target_depth = o3d.io.read_image(depth_dir + '{}.png'.format(target_num))

# set pinhole camera matrix
config = Config()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fx, config.fy, config.cx, config.cy)

source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_rgb, source_depth)
target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_rgb, target_depth)

source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)

source_temp = copy.deepcopy(source_pcd)
target_temp = copy.deepcopy(target_pcd)
source_temp.transform(Rs)

ans_pcd = o3d.geometry.PointCloud()
source_points = np.asarray(source_temp.points)
target_points = np.asarray(target_temp.points)

source_colors = np.asarray(source_temp.colors)
target_colors = np.asarray(target_temp.colors)

points = np.concatenate([source_points, target_points], axis=0)
colors = np.concatenate([source_colors, target_colors], axis=0)

ans_pcd.points = o3d.utility.Vector3dVector(points)
ans_pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud('source.ply', source_pcd)
o3d.io.write_point_cloud('target.ply', target_pcd)
o3d.io.write_point_cloud('ans.ply', ans_pcd)
