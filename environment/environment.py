import argparse
import yaml
import torch
import os
import numpy as np

from gaussian_loader import GaussianLoader

from gsplat.rendering import rasterization
from scipy.spatial.transform import Rotation as Rotation
from matplotlib import pyplot as plt


class Environment:
    def __init__(self, args):

        os.environ["PATH"] = f'/usr/local/cuda/bin:{os.environ["PATH"]}'

        self.ply_path = args["ply_path"]
        self.device = args["device"]
        self.width = args["width"]
        self.height = args["height"]

        gaussian_loader = GaussianLoader(self.ply_path, self.device)
        self.gaussian_data = gaussian_loader.load_gaussian_splat()

        # Initialize pos and rot
        self.viewmats = None
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x, z, y, roll, pitch, yaw
        self.update_state(action=np.array([0.0, -15.0, 2.0, 90.0, 0.0, 0.0]))


    def step(self, action):
        self.update_state(action)
        state = self.render_state()
        reward = self.calculate_reward()
        return state, reward

    def update_state(self, action):
        self.state = self.state + action
        pos = self.state[0:3]
        rot = self.state[3:6]
        rot_mat, translation_vector = self.convert_to_colmap_pose(pos=pos, rot=rot)
        viewmats = torch.zeros(1, 4, 4)
        viewmats[0, 0:3, 0:3] = torch.tensor(rot_mat)
        viewmats[0, 3:4, 0:3] = torch.tensor(translation_vector).unsqueeze(0)
        viewmats[0, 3, 3] = 1.0
        self.viewmats = viewmats.permute(0, 2, 1).to(self.device)

    def render_state(self):
        Ks = torch.tensor([[[self.width/3, 0.0, self.width/2], [0.0, self.width/3, self.height/2], [0., 0., 1.]]], device=self.device)
        color, alpha, meta = rasterization(viewmats=self.viewmats, Ks=Ks, width=self.width, height=self.height, sh_degree=3, **self.gaussian_data)
        return color

    @staticmethod
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

    def calculate_reward(self):
        return 0.0

    def reset(self):
        pass

    def visualize(self):
        color = self.render_state()
        image_np = color.squeeze(0).cpu().detach().numpy()
        height, width = image_np.shape[:2]
        aspect_ratio = width / height
        n = 5  # Multiply by a factor to control the size
        fig = plt.figure(figsize=(aspect_ratio * n, n))
        plt.imshow(image_np, aspect='auto')
        plt.axis('off')
        plt.show()


def main(config):

    environment = Environment(config)
    num_step = 100

    for i in range(num_step):
        environment.visualize()
        environment.update_state([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--configs", default="./configs/default.yaml", help="path to default.yaml file")
    args = argParser.parse_args()

    with open(args.configs, "r") as stream:
        args = yaml.safe_load(stream)

    main(args)