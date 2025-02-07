try:
    from gaussian_wrangler.paths_local import PATH_GS
except ImportError:
    raise ImportError(
        "Please specify the path to the gaussian-splatting repository via PATH_GS in paths_local.py"
    )

try:
    from gaussian_renderer import render
    from scene.cameras import Camera
    from gaussian_renderer import GaussianModel
    from arguments import PipelineParams
    from argparse import ArgumentParser
except ImportError:
    import sys

    sys.path.append(PATH_GS)
    from gaussian_renderer import render
    from scene.cameras import Camera
    from gaussian_renderer import GaussianModel
    from arguments import PipelineParams
    from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import Parameter


def gaussians_from_np(gaussian_array: np.ndarray):
    gaussians = GaussianModel(sh_degree=1)

    xyz = gaussian_array[:, :3]
    gaussians._xyz = Parameter(
        torch.tensor(xyz, dtype=torch.float32).requires_grad_(True)
    ).to("cuda")

    features_dc = gaussian_array[:, None, 3:6]
    gaussians._features_dc = Parameter(
        torch.tensor(features_dc, dtype=torch.float32).requires_grad_(True)
    ).to("cuda")

    features_rest = gaussian_array[:, 6:15].reshape(-1, 3, 3)
    gaussians._features_rest = Parameter(
        torch.tensor(features_rest, dtype=torch.float32).requires_grad_(True)
    ).to("cuda")

    opacity = gaussian_array[:, 15:16]
    gaussians._opacity = Parameter(
        torch.tensor(opacity, dtype=torch.float32).requires_grad_(True)
    ).to("cuda")

    if gaussian_array.shape[1] == 22:
        scaling = gaussian_array[:, 16:18]
        scaling = np.hstack([scaling, np.ones((scaling.shape[0], 1)) * -5])
    elif gaussian_array.shape[1] == 23:
        scaling = gaussian_array[:, 16:19]
    else:
        raise ValueError(f"Unsupported number of components: {gaussian_array.shape[1]}")
    gaussians._scaling = Parameter(
        torch.tensor(scaling, dtype=torch.float32).requires_grad_(True)
    ).to("cuda")

    rotation = gaussian_array[:, -4:]
    gaussians._rotation = Parameter(
        torch.tensor(rotation, dtype=torch.float32).requires_grad_(True)
    ).to("cuda")

    gaussians.active_sh_degree = 1

    return gaussians
