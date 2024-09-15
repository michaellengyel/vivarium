import torch
from plyfile import PlyData
from gsplat.rendering import rasterization
from gsplat import project_gaussians, rasterize_gaussians
import numpy as np
import os
from matplotlib import pyplot as plt

from scipy.spatial.transform import Rotation as Rotation

def convert_to_colmap_pose(pos, rot):
    # Blender to colmap rotation matrix
    rot_mat_b2c = Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()

    # Convert Euler angles to a rotation matrix using the 'xyz' order
    rot_mat_blender = Rotation.from_euler('xyz', rot, degrees=True).as_matrix()
    rot_mat = rot_mat_b2c @ rot_mat_blender.T

    # Compute the translation vector
    translation_vector = np.dot(-rot_mat, pos)

    # Return the pose in the desired format
    return rot_mat.T, translation_vector

os.environ["PATH"] = f'/usr/local/cuda/bin:{os.environ["PATH"]}'
device = "cuda:0"

ply_path = '/home/peter/work/nerfstudio/exports/splat/splat.ply'
ply_path = '/home/peter/PycharmProjects/vivarium/scenes/scene_0/splat/point_cloud/iteration_30000/point_cloud.ply'
ply_data = PlyData.read(ply_path)
ply_data = ply_data['vertex'].data

num_coeffs = 16
coeff_idxs = []
for i in range(num_coeffs-1):
    for j in range(3):
        coeff_idxs.append(i+j*(num_coeffs-1))

means = torch.tensor(np.vstack([ply_data['x'], ply_data['y'], ply_data['z']])).to(device).permute(1, 0)
scales = torch.tensor(np.vstack([ply_data['scale_0'], ply_data['scale_1'], ply_data['scale_2']])).to(device).permute(1, 0)  # / -10000
quats = torch.tensor(np.vstack([ply_data['rot_0'], ply_data['rot_1'], ply_data['rot_2'], ply_data['rot_3']])).to(device).permute(1, 0)
opacities = torch.tensor(ply_data['opacity']).to(device)
colors = torch.tensor(np.vstack([ply_data['f_dc_0'], ply_data['f_dc_1'], ply_data['f_dc_2']] + [ply_data[f"f_rest_{coeff_idxs[i]}"] for i in range(3 * (num_coeffs-1))])).to(device).permute(1, 0).reshape(-1, num_coeffs, 3)

# Adding target gaussian
t_mean = torch.tensor([[2, 0, 2]])
t_scale = torch.tensor([[-3.0, -3.0, -3.0]])
t_scale = torch.tensor([[1.0, 1.0, 1.0]]) * -1.0
t_quat = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
t_opacities = torch.tensor([[2.7]])
t_colors = torch.zeros((1, 16, 3))
t_colors[:, 0, 0] = 2.8

means[0:1, :] = t_mean
scales[0:1, :] = t_scale
quats[0:1, :] = t_quat
opacities[0:1] = t_opacities
colors[0:1, :, :] = t_colors

n = 5

pos = np.array([0.0, -15.0, 2.0])
rot = np.array([90.0, 0.0, -20.0])
rot_mat, translation_vector = convert_to_colmap_pose(pos, rot)
viewmats = torch.zeros(1, 4, 4)
viewmats[0, 0:3, 0:3] = torch.tensor(rot_mat)
viewmats[0, 3:4, 0:3] = torch.tensor(translation_vector).unsqueeze(0)
viewmats[0, 3, 3] = 1.0
viewmats = viewmats.to(device).permute(0, 2, 1)

for i in range(1000):

    # t_mean = torch.tensor([[0, 0.001, 0]]).to(device)
    # means[0:1, :] += t_mean

    # define cameras
    #viewmats = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0],
    #                          [0.0, 0.0, -1.0, i],  # i was here
    #                          [0.0, 0.0, 0.0, 1.0]]], device=device)

    Ks = torch.tensor([[300.*n, 0., 150.*n], [0., 300.*n, 100.*n], [0., 0., 1.]], device=device)[None, :, :]

    width, height = 300*n, 200*n
    color, alpha, meta = rasterization(means, quats, torch.exp(scales), torch.sigmoid(opacities), colors, viewmats, Ks, width, height, sh_degree=3)
    print(color.shape)
    print(alpha.shape)

    # Convert the tensor to a numpy array and remove the batch dimension
    image_np = color.squeeze(0).cpu().detach().numpy()
    # image_np = alpha.squeeze(0).cpu().detach().numpy()

    # Visualize the image
    plt.imshow(image_np, aspect='auto')
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
