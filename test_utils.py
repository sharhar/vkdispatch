import numpy as np
import test_correlator as tc

import typing

def load_coords(file_path):
    atom_data = np.load(file_path)
    atom_coords: np.ndarray = atom_data["coords"].astype(np.float32)
    #atom_proton_counts: np.ndarray = atom_data["proton_counts"].astype(np.float32)

    atom_coords -= np.sum(atom_coords, axis=0) / atom_coords.shape[0]

    return atom_coords

def load_image(file_path):
    input_image_raw: np.ndarray = np.load(file_path)

    n_std = 5
    outliers_idxs = np.abs(input_image_raw - input_image_raw.mean()) > n_std * input_image_raw.std()
    input_image_raw[outliers_idxs] = input_image_raw.mean()
    test_image_normalized: np.ndarray = (input_image_raw - input_image_raw.mean()) / input_image_raw.std()

    return np.fft.fftshift(test_image_normalized)

def get_rotation_matrix(angles: typing.List[int], offsets: typing.List[int] = [0, 0]):
    in_matricies = np.zeros(shape=(4, 4), dtype=np.float32)

    cos_phi   = np.cos(np.deg2rad(angles[0]))
    sin_phi   = np.sin(np.deg2rad(angles[0]))
    cos_theta = np.cos(np.deg2rad(angles[1]))
    sin_theta = np.sin(np.deg2rad(angles[1]))

    M00 = cos_phi * cos_theta 
    M01 = -sin_phi 

    M10 = sin_phi * cos_theta 
    M11 = cos_phi 

    M20 = -sin_theta 

    cos_psi_in_plane   = np.cos(np.deg2rad(-angles[2] - 90)) 
    sin_psi_in_plane   = np.sin(np.deg2rad(-angles[2] - 90))

    m00  = cos_psi_in_plane
    m01 = sin_psi_in_plane
    m10 = -sin_psi_in_plane
    m11 = cos_psi_in_plane

    in_matricies[0, 0] = m00 * M00 + m10 * M01
    in_matricies[0, 1] = m00 * M10 + m10 * M11
    in_matricies[0, 2] = m00 * M20
    in_matricies[0, 3] = offsets[0]
    
    in_matricies[1, 0] = m01 * M00 + m11 * M01
    in_matricies[1, 1] = m01 * M10 + m11 * M11
    in_matricies[1, 2] = m01 * M20
    in_matricies[1, 3] = offsets[1]

    return in_matricies.T

def aggregate_results(correlator: "tc.Correlator"):
    max_crosses = correlator.max_cross.read()
    best_indicies = correlator.best_index.read()

    final_results = [np.zeros(shape=(correlator.shift_buffer.shape[0], correlator.shift_buffer.shape[1], 2), dtype=np.float64) for _ in max_crosses]

    for i in range(len(max_crosses)):
        final_results[i][:, :, 0] = max_crosses[i]
        final_results[i][:, :, 1] = best_indicies[i]

    true_final_result = final_results[0]

    for other_result in final_results[1:]:
        true_final_result = np.where(other_result[:, :, 0:1] > true_final_result[:, :, 0:1], other_result, true_final_result)

    final_max_cross = np.fft.ifftshift(true_final_result[:, :, 0])
    best_index_result = np.fft.ifftshift(true_final_result[:, :, 1].astype(np.int32))

    index_of_max = np.unravel_index(np.argmax(final_max_cross), final_max_cross.shape)
    final_index = best_index_result[index_of_max]

    return final_max_cross, best_index_result, index_of_max, final_index