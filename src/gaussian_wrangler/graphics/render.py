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

import torch
import numpy as np
from PIL import Image
import json
from gaussian_wrangler.file_handling.np_to_gaussian import gaussians_from_np


def default_camera():
    RT_cam = np.array(
        [
            [-0.85064932, 0.0, -0.52573352, 1.28860167],
            [0.52573352, 0.0, -0.85064932, 1.77597621],
            [0.0, -1.0, -0.0, 0.5],
            [-0.0, -0.0, -0.0, 1.0],
        ]
    )
    extr = np.linalg.inv(RT_cam)
    cam = Camera(
        resolution=(400, 400),
        colmap_id=0,
        R=RT_cam[:3, :3],
        T=extr[:3, 3],
        FoVx=1.0,
        FoVy=1.0,
        depth_params=None,
        image=Image.new("RGB", (400, 400)),
        invdepthmap=None,
        image_name="0.png",
        uid=0,
    )
    return cam


def camera_from_dict(camera):
    assert "R" in camera
    assert "T" in camera
    assert "FoVx" in camera
    assert "FoVy" in camera
    assert "resolution" in camera

    cam = Camera(
        resolution=camera["resolution"],
        colmap_id=0,
        R=camera["R"],
        T=camera["T"],
        FoVx=camera["FoVx"],
        FoVy=camera["FoVy"],
        depth_params=None,
        image=Image.new("RGB", camera["resolution"]),
        invdepthmap=None,
        image_name="0.png",
        uid=0,
    )
    return cam


def render_image(gaussians, out_path=None, camera=None, sh_degree=1):
    # 1. Setup camera
    if camera is not None:
        if isinstance(camera, Camera):
            print("Using provided camera")
            cam = camera
        elif isinstance(camera, dict):
            print("Loading camera from dictionary")
            cam = camera_from_dict(camera)
        else:
            raise ValueError(
                f"Invalid camera type: {type(camera)}. Must be Camera or dict"
            )
    else:
        print("Using default camera")
        cam = default_camera()

    # 2. Load gaussians
    if isinstance(gaussians, GaussianModel):
        print("Using provided GaussianModel")
        gaussians = gaussians
    elif isinstance(gaussians, str):
        print("Loading GaussianModel from path")
        gaussian_path = gaussians
        gaussians = GaussianModel(sh_degree=sh_degree)
        gaussians.load_ply(path=gaussian_path)
    elif isinstance(gaussians, np.ndarray):
        print("Loading Gaussians from numpy array")
        gaussians = gaussians_from_np(gaussians)

    if gaussians._scaling.shape[1] == 2:
        gaussians._scaling = torch.cat(
            [gaussians._scaling, torch.ones_like(gaussians._scaling[:, :1]) * -10],
            dim=1,
        )  # Add third dimension to scaling

    print(
        gaussians._xyz.shape,
        gaussians._features_dc.shape,
        gaussians._features_rest.shape,
        gaussians._opacity.shape,
        gaussians._scaling.shape,
        gaussians._rotation.shape,
    )

    # 3. Setup background color
    background = torch.zeros(3).to("cuda")

    # 4. Setup pipeline
    argparser2 = ArgumentParser(description="Testing script parameters")
    pipeline = PipelineParams(argparser2)

    rendering = render(
        cam, gaussians, pipeline, background, use_trained_exp=False, separate_sh=False
    )["render"]
    print("rendered")

    # 5. Save rendering
    if out_path is not None:
        rendering_np = (
            (rendering.permute((1, 2, 0)) * 255).to(torch.uint8).detach().cpu().numpy()
        )
        print(rendering_np.shape)
        img = Image.fromarray(rendering_np)
        img.save(out_path)

    return rendering


if __name__ == "__main__":
    parser = ArgumentParser(description="Render image")
    parser.add_argument("--gaussian_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    render_image(args.gaussian_path, args.out_path)
