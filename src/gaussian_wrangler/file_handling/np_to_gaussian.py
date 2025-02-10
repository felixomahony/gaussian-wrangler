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


def gaussians_from_np(gaussian_array: np.ndarray | torch.Tensor) -> GaussianModel:
    """
    Convert a numpy array to a GaussianModel object.

    Args:
        gaussian_array: A numpy array of shape (n, 22) for 2d Gaussians or (n, 23) for 3d Gaussians where n is the number of Gaussians.

    Returns:
        A GaussianModel object.
    """

    if isinstance(gaussian_array, np.ndarray):
        gaussian_array = torch.tensor(gaussian_array, dtype=torch.float32).to("cuda")

    gaussians = GaussianModel(sh_degree=1)

    xyz = gaussian_array[:, :3]
    gaussians._xyz = xyz

    features_dc = gaussian_array[:, None, 3:6]
    gaussians._features_dc = features_dc

    features_rest = gaussian_array[:, 6:15].reshape(-1, 3, 3)
    gaussians._features_rest = features_rest

    opacity = gaussian_array[:, 15:16]
    gaussians._opacity = opacity

    if gaussian_array.shape[1] == 22:
        scaling = gaussian_array[:, 16:18]
        scaling = torch.hstack([scaling, torch.ones_like(scaling[:, 0:1]) * -5])
    elif gaussian_array.shape[1] == 23:
        scaling = gaussian_array[:, 16:19]
    else:
        raise ValueError(f"Unsupported number of components: {gaussian_array.shape[1]}")
    gaussians._scaling = scaling

    rotation = gaussian_array[:, -4:]
    gaussians._rotation = rotation

    gaussians.active_sh_degree = 1

    return gaussians
