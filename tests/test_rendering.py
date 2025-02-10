from gaussian_wrangler.graphics.render import render_image

import numpy as np
from PIL import Image
import torch


def test_render():
    gaussian = np.ones((1, 23))
    gaussian[:, 16:19] = -2
    gaussian[:, :3] = 0.5

    rendering = render_image(gaussian)
    im = Image.fromarray(
        (rendering.permute((1, 2, 0)) * 255).to(torch.uint8).detach().cpu().numpy()
    )

    im.save("a.png")


def test_render_torch():
    gaussian = torch.ones((1, 23)).to("cuda")
    gaussian[:, 16:19] = -2
    gaussian[:, :3] = 0.5

    rendering = render_image(gaussian)
    im = Image.fromarray(
        (rendering.permute((1, 2, 0)) * 255).to(torch.uint8).detach().cpu().numpy()
    )

    im.save("a.png")
