import vkdispatch as vd

vd.initialize(debug_mode=True)
vd.make_context(use_cpu=True)

import numpy as np

def test_1d_buffer_creation():
    # Create a 1D buffer
    signal = np.sin(np.array([i/8 for i in range(0, 50, 1)])).astype(np.float32)

    test_line = vd.Buffer((len(signal), ), vd.float32)
    test_line.write(signal)

    assert np.allclose(test_line.read(0), signal)

def test_2d_buffer_creation():
    # Create a 2D buffer
    signal_2d = np.sin(np.array([[i/8 + j/17 for i in range(0, 50, 1)] for j in range(0, 50, 1)])).astype(np.float32)

    test_img = vd.Buffer(signal_2d.shape, vd.float32)
    test_img.write(signal_2d)

    assert np.allclose(test_img.read(0), signal_2d)

def test_3d_buffer_creation():
   # Create a 3D buffer
    signal_3d = np.sin(np.array([[[i/8 + j/17 + k/23 for i in range(0, 50, 1)] for j in range(0, 50, 1)] for k in range(0, 50, 1)])).astype(np.float32)

    test_img = vd.Buffer(signal_3d.shape, vd.float32)
    test_img.write(signal_3d)

    assert np.allclose(test_img.read(0), signal_3d)


def test_1d_buffer_vec2_creation():
    # Create a 1D buffer
    signal = np.sin(np.array([i/8 for i in range(0, 50, 1)])).astype(np.float32)
    signal_vec2 = np.array([signal, signal * np.cos([(i + 56)/7 for i in range(0, 50, 1)])]).astype(np.float32)

    test_line = vd.asbuffer(signal_vec2)

    assert np.allclose(test_line.read(0), signal_vec2)

def test_2d_buffer_vec2_creation():
    # Create a 2D buffer
    signal_2d = np.sin(np.array([[i/8 + j/17 for i in range(0, 50, 1)] for j in range(0, 50, 1)])).astype(np.float32)
    signal_vec2 = np.array([signal_2d, signal_2d * np.cos([[i/8 + j/17 for i in range(0, 50, 1)] for j in range(0, 50, 1)])]).astype(np.float32).T

    test_img = vd.Buffer(signal_vec2.shape[:2], vd.vec2)
    test_img.write(signal_vec2)

    assert np.allclose(test_img.read(0), signal_vec2)

def test_3d_buffer_vec2_creation():
    # Create a 3D buffer
    signal_3d = np.sin(np.array([[[i/8 + j/17 + k/23 for i in range(0, 50, 1)] for j in range(0, 50, 1)] for k in range(0, 50, 1)])).astype(np.float32)
    signal_vec2 = np.array([signal_3d, signal_3d * np.cos([[[i/8 + j/17 + k/23 for i in range(0, 50, 1)] for j in range(0, 50, 1)] for k in range(0, 50, 1)])]).astype(np.float32).T

    test_img = vd.Buffer(signal_vec2.shape[:3], vd.vec2)
    test_img.write(signal_vec2)

    assert np.allclose(test_img.read(0), signal_vec2)