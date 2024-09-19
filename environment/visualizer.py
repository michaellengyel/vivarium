from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import os
import numpy as np

class Visualizer:
    def __init__(self):
        self.states = []

    def append_state(self, state):
        self.states.append(state)

    def visualize(self, idx, color, reward=0, plot=True):

        image_np = color.squeeze(0).cpu().detach().numpy()

        if plot:
            height, width = image_np.shape[:2]
            aspect_ratio = width / height
            n = 5  # Multiply by a factor to control the size
            plt.figure(figsize=(aspect_ratio * n, n))
            plt.imshow(image_np, aspect='auto')
            plt.axis('off')
            plt.show()

        else:
            image = Image.fromarray((image_np * 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), str(reward))
            path = "./output/"
            if not os.path.exists(path):
                os.makedirs(path)
            image.save(path + str(idx).zfill(8) + ".png")


    def visualize_top(self):
        plt.scatter()
        plt.show()