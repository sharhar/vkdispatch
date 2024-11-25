import vkdispatch as vd

vd.make_context(use_cpu=True)

import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import numpy as np

def test_1d_image_creation():
    # Create a 1D image
    signal = np.sin(np.array([i/8 for i in range(0, 50, 1)])).astype(np.float32)

    test_line = vd.Image1D(len(signal), vd.float32)
    test_line.write(signal)

    assert np.allclose(test_line.read(0), signal)

def test_2d_image_creation():
    # Create a 2D image
    signal_2d = np.sin(np.array([[i/8 + j/17 for i in range(0, 50, 1)] for j in range(0, 50, 1)])).astype(np.float32)

    test_img = vd.Image2D(signal_2d.shape, vd.float32)
    test_img.write(signal_2d)

    assert np.allclose(test_img.read(0), signal_2d)

def test_3d_image_creation():
    # Create a 3D image
    signal_3d = np.sin(np.array([[[i/8 + j/17 + k/23 for i in range(0, 50, 1)] for j in range(0, 50, 1)] for k in range(0, 50, 1)])).astype(np.float32)

    test_img = vd.Image3D(signal_3d.shape, vd.float32)
    test_img.write(signal_3d)

    assert np.allclose(test_img.read(0), signal_3d)

def test_1d_image_linear_sampling():
    # Create a 1D image
    signal = np.sin(np.array([i/8 for i in range(0, 50, 1)])).astype(np.float32)
    sample_factor = 10

    test_line = vd.Image1D(len(signal), vd.float32)
    test_line.write(signal)

    result_arr = vd.Buffer((len(signal) * (sample_factor - 1),), vd.float32)

    @vd.shader(exec_size=lambda args: args.buff.size)
    def do_approx(buff: Buff[f32], line: Img1[f32]):
        ind = vc.global_invocation().x.copy()
        buff[ind] = line.sample((ind.cast_to(f32)) / sample_factor).x

    do_approx(result_arr, test_line)

    signal_full = np.sin(np.array([i/80 for i in range(0, 450, 1)])).astype(np.float32)

    assert np.allclose(result_arr.read()[0], signal_full, atol=0.002)

def test_2d_image_linear_sampling():
    # Create a 2D image
    signal_2d = np.sin(np.array([[i/8 + j/17 for i in range(0, 50, 1)] for j in range(0, 50, 1)])).astype(np.float32)
    sample_factor = 10

    test_img = vd.Image2D(signal_2d.shape, vd.float32)
    test_img.write(signal_2d)

    result_arr = vd.Buffer((signal_2d.shape[0] * (sample_factor - 1), signal_2d.shape[1] * (sample_factor - 1)), vd.float32)

    @vd.shader(exec_size=lambda args: args.buff.size)
    def do_approx(buff: Buff[f32], img: Img2[f32]):
        ind = vc.global_invocation().x.copy()
        ind_2d = vc.unravel_index(ind, buff.shape)
        buff[ind] = img.sample((ind_2d.cast_to(v2)) / sample_factor).x

    do_approx(result_arr, test_img)

    signal_full = np.sin(np.array([[i/80 + j/170 for i in range(0, 450, 1)] for j in range(0, 450, 1)])).astype(np.float32)

    assert np.allclose(result_arr.read()[0], signal_full, atol=0.0025)

# def test_3d_image_linear_sampling():
#     # Create a 3D image
#     signal_3d = np.sin(np.array([[[i/8 + j/17 + k/23 for i in range(0, 5, 1)] for j in range(0, 5, 1)] for k in range(0, 5, 1)]).astype(np.float32))
#     sample_factor = 10

#     test_img = vd.Image3D(signal_3d.shape, vd.float32)
#     test_img.write(signal_3d)

#     result_arr = vd.Buffer((signal_3d.shape[0] * (sample_factor - 1), signal_3d.shape[1] * (sample_factor - 1), signal_3d.shape[2] * (sample_factor - 1)), vd.float32)

#     @vc.shader(exec_size=lambda args: args.buff.size)
#     def do_approx(buff: Buff[f32], img: Img3[f32]):
#         ind = vc.global_invocation.x.copy()
#         ind_3d = vc.unravel_index(ind, buff.shape)
#         buff[ind] = img.sample((ind_3d.cast_to(v3)) / sample_factor).x

#     do_approx(result_arr, test_img)

#     signal_full = np.sin(np.array([[[i/80 + j/170 + k/230 for i in range(0, 45, 1)] for j in range(0, 45, 1)] for k in range(0, 45, 1)]).astype(np.float32))

#     assert np.allclose(result_arr.read()[0], signal_full, atol=0.01)