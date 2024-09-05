import numpy as np
import torch
from plyfile import PlyData


class GaussianLoader:

    def __init__(self, ply_path, device):
        ply_data = PlyData.read(ply_path)
        ply_data = ply_data['vertex'].data

        num_coeffs = 16
        coeff_idxs = []
        for i in range(num_coeffs - 1):
            for j in range(3):
                coeff_idxs.append(i + j * (num_coeffs - 1))

        means = torch.tensor(np.vstack([ply_data['x'], ply_data['y'], ply_data['z']])).to(device).permute(1, 0)
        scales = torch.tensor(np.vstack([ply_data['scale_0'], ply_data['scale_1'], ply_data['scale_2']])).to(device).permute(1, 0)
        scales = torch.exp(scales)
        quats = torch.tensor(np.vstack([ply_data['rot_0'], ply_data['rot_1'], ply_data['rot_2'], ply_data['rot_3']])).to(device).permute(1, 0)
        opacities = torch.tensor(ply_data['opacity']).to(device)
        opacities = torch.sigmoid(opacities)
        colors = torch.tensor(np.vstack([ply_data['f_dc_0'], ply_data['f_dc_1'], ply_data['f_dc_2']] + [ply_data[f"f_rest_{coeff_idxs[i]}"] for i in range(3 * (num_coeffs - 1))])).to(device).permute(1, 0).reshape(-1, num_coeffs, 3)

        self.data = {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "colors": colors}

    def load_gaussian_splat(self):
        return self.data