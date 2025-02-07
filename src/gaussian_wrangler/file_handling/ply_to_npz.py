import numpy as np
from tqdm import tqdm
import plyfile


def ply_to_npz(path, n_voxels=125):
    splat = plyfile.PlyData.read(path)

    assert len(splat.elements) == 1
    assert splat.elements[0].name == "vertex"
    assert len(splat.elements[0].data) > 0

    full_splats = True
    if len(splat.elements[0].data[0]) != 26:
        full_splats = False
    elif len(splat.elements[0].data[0]) != 25:
        raise ValueError("Unsupported number of components")

    splat_matrix = np.zeros((len(splat.elements[0].data), 23))
    # 23 components
    # x, y, z, f_dc_0, f_dc_1, f_dc_2, f_rest_0..8, opacity, scale_0..2, rot_0..3

    # xyz
    splat_matrix[:, 0] = splat.elements[0].data["x"]
    splat_matrix[:, 1] = splat.elements[0].data["y"]
    splat_matrix[:, 2] = splat.elements[0].data["z"]

    # f_dc
    splat_matrix[:, 3] = splat.elements[0].data["f_dc_0"]
    splat_matrix[:, 4] = splat.elements[0].data["f_dc_1"]
    splat_matrix[:, 5] = splat.elements[0].data["f_dc_2"]

    # f_rest
    for i in range(9):
        splat_matrix[:, 6 + i] = splat.elements[0].data[f"f_rest_{i}"]

    # opacity
    splat_matrix[:, 15] = splat.elements[0].data["opacity"]

    # scale
    splat_matrix[:, 16] = splat.elements[0].data["scale_0"]
    splat_matrix[:, 17] = splat.elements[0].data["scale_1"]
    if full_splats:
        splat_matrix[:, 18] = splat.elements[0].data["scale_2"]

    # rot
    splat_matrix[:, 19] = splat.elements[0].data["rot_0"]
    splat_matrix[:, 20] = splat.elements[0].data["rot_1"]
    splat_matrix[:, 21] = splat.elements[0].data["rot_2"]
    splat_matrix[:, 22] = splat.elements[0].data["rot_3"]

    if not full_splats:
        splat_matrix = np.hstack(
            [
                splat_matrix[:, :18],
                splat_matrix[:, 19:],
            ],
        )

    sz = splat_matrix.shape[0]
    splat_matrix = clip(splat_matrix)
    print(f"Clipped {sz - splat_matrix.shape[0]} splats")

    # check if voxelisation is possible
    voxel_indices = np.floor(splat_matrix[:, :3] * n_voxels).astype(int)
    voxel_centers = (voxel_indices + 0.5) / n_voxels

    # check if voxel_indices are unique
    print("Total number of voxels:", len(voxel_indices))
    print(
        "Number of overlapping voxels:",
        len(voxel_indices) - len(np.unique(voxel_indices, axis=0)),
    )

    splat_matrix[:, :3] -= voxel_centers
    splat_matrix[:, :3] *= n_voxels

    splat_matrix, voxel_indices = set_voxels(splat_matrix, voxel_indices, n_voxels)

    return splat_matrix, voxel_indices


def set_voxels(splat_matrix, voxel_indices, n_voxels):
    n_splats = splat_matrix.shape[0]
    mask = np.ones(n_splats, dtype=bool)
    n_fixed = 0
    for v in range(n_splats):
        voxels = voxel_indices[v]
        if np.any(np.all(voxel_indices[v + 1 :] == voxels[None], axis=1)):
            offsets = splat_matrix[v, :3]
            offsets_signed = np.sign(offsets)
            fix_found = False
            for i, j, k in np.ndindex(2, 2, 2):
                if i == j == k == 0:
                    continue
                ijk = np.array([i, j, k])
                if (
                    not np.any(
                        np.all(
                            voxel_indices[v + 1 :]
                            == (voxels + offsets_signed * ijk)[None],
                            axis=1,
                        )
                    )
                    and not np.any(
                        np.all(
                            voxel_indices[:v] == (voxels + offsets_signed * ijk)[None],
                            axis=1,
                        )
                    )
                    and not np.any((voxels + offsets_signed * ijk) < 0)
                    and not np.any((voxels + offsets_signed * ijk) >= n_voxels)
                ):
                    splat_matrix[v, :3] += -offsets_signed * ijk + splat_matrix[v, :3]
                    voxel_indices[v] = voxels + offsets_signed * ijk
                    fix_found = True
                    n_fixed += 1
                    break

            if not fix_found:
                mask[v] = False

    print("Fixed", n_fixed, "splats")
    print(f"Removed {np.sum(~mask)} overlapping splats")
    print(f"Total number of splats: {np.sum(mask)}")
    return splat_matrix[mask], voxel_indices[mask]


def clip(splat_matrix):
    clip_mask = np.any(np.abs(splat_matrix[:, :3] - 0.5) > 0.5, axis=1)

    return splat_matrix[~clip_mask]
