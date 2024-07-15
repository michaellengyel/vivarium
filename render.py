from matplotlib import pyplot as plt
from gsplat.rendering import rasterization
import torch
import os

# print(os.environ)
print("LD_LIBRARY_PATH", os.environ["LD_LIBRARY_PATH"])
print(os.environ["PATH"])
print(torch.cuda.is_available())

device = "cuda:0"
means = torch.randn((100, 3), device=device)
quats = torch.randn((100, 4), device=device)
scales = torch.rand((100, 3), device=device) * 0.1
colors = torch.rand((100, 3), device=device)
opacities = torch.rand((100,), device=device)

# define cameras
viewmats = torch.eye(4, device=device)[None, :, :]

Ks = torch.tensor([[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]

width, height = 300, 200
colors, alphas, meta = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)
print(colors.shape)
print(alphas.shape)

# Convert the tensor to a numpy array and remove the batch dimension
image_np = colors.squeeze(0).cpu().detach().numpy()

# Visualize the image
plt.imshow(image_np)
plt.title("Sample RGB Image of Size (200x300)")
plt.axis('off')  # Hide axes for better visualization
plt.show()

print("Working...")
