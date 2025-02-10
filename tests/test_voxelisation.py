from gaussian_wrangler.file_handling.ply_to_npz import set_voxels, voxels_to_xyz
import numpy as np
import matplotlib.pyplot as plt


def test_voxelisation():
    splat_matrix = np.array([[0.2, 0.1, 0.4], [0.6, 0.1, 0.4]])

    voxel_offsets, voxel_indices = set_voxels(splat_matrix, 2)

    splat_matrix_reformed = voxels_to_xyz(voxel_offsets.copy(), voxel_indices.copy(), 2)

    print(splat_matrix)
    print(splat_matrix_reformed)

    assert np.allclose(splat_matrix, splat_matrix_reformed)


def test_voxel_setting():
    splat_matrix = np.array([[0.2, 0.1, 0.4], [0.3, 0.1, 0.4], [0.3, 0.1, 0.4]])

    voxel_offsets, voxel_indices = set_voxels(splat_matrix, 2)

    splat_matrix_reformed = voxels_to_xyz(voxel_offsets.copy(), voxel_indices.copy(), 2)

    print(splat_matrix)
    print(splat_matrix_reformed)

    assert np.allclose(splat_matrix, splat_matrix_reformed)


def test_voxel_culling():
    splat_matrix = np.ones((9, 3)) * 0.4

    _, voxel_indices = set_voxels(splat_matrix, 2)

    print(splat_matrix)
    print(voxel_indices)

    assert voxel_indices.shape[0] < splat_matrix.shape[0]
