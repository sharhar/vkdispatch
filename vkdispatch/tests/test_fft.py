import vkdispatch as vd

import numpy as np

def make_random_complex_signal(shape):
    r = np.random.random(size=shape)
    i = np.random.random(size=shape)
    return (r + i * 1j).astype(np.complex64)

def test_fft_1d():
    # Create a 1D buffer
    signal = make_random_complex_signal((50,))

    test_line = vd.Buffer((signal.shape[0],), vd.complex64)
    test_line.write(signal)

    # Perform an FFT on the buffer
    vd.fft(test_line)

    assert np.allclose(test_line.read(0), np.fft.fft(signal), atol=0.00001) or np.allclose(test_line.read(0), np.fft.fft(signal) * np.prod(signal.shape), atol=0.00001)

def test_fft_2d():
    # Create a 2D buffer
    signal_2d = make_random_complex_signal((50, 50))

    test_img = vd.Buffer(signal_2d.shape, vd.complex64)
    test_img.write(signal_2d)

    # Perform an FFT on the buffer
    vd.fft(test_img)

    assert np.allclose(test_img.read(0), np.fft.fft2(signal_2d), atol=0.0001) or np.allclose(test_img.read(0), np.fft.fft2(signal_2d) * np.prod(signal_2d.shape), atol=0.0001)

def test_fft_3d():
    # Create a 3D buffer
    signal_3d = make_random_complex_signal((50, 50, 50))

    test_img = vd.Buffer(signal_3d.shape, vd.complex64)
    test_img.write(signal_3d)

    # Perform an FFT on the buffer
    vd.fft(test_img)

    assert np.allclose(test_img.read(0), np.fft.fftn(signal_3d), atol=0.01) or np.allclose(test_img.read(0), np.fft.fftn(signal_3d) * np.prod(signal_3d.shape), atol=0.01)

def test_ifft_1d():
    # Create a 1D buffer
    signal = make_random_complex_signal((50, ))

    test_line = vd.Buffer((signal.shape[0],), vd.complex64)
    test_line.write(signal)

    # Perform an IFFT on the buffer
    vd.ifft(test_line)
    
    assert np.allclose(test_line.read(0), np.fft.ifft(signal) * np.prod(signal.shape), atol=0.00001) or np.allclose(test_line.read(0), np.fft.ifft(signal), atol=0.00001)

def test_ifft_2d():
    # Create a 2D buffer
    signal_2d = make_random_complex_signal((50, 50))

    test_img = vd.Buffer(signal_2d.shape, vd.complex64)
    test_img.write(signal_2d)

    # Perform an IFFT on the buffer
    vd.ifft(test_img)

    assert np.allclose(test_img.read(0), np.fft.ifft2(signal_2d) * np.prod(signal_2d.shape), atol=0.0001) or np.allclose(test_img.read(0), np.fft.ifft2(signal_2d), atol=0.0001)

def test_ifft_3d():
    # Create a 3D buffer
    signal_3d = make_random_complex_signal((50, 50, 50))

    test_img = vd.Buffer(signal_3d.shape, vd.complex64)
    test_img.write(signal_3d)

    # Perform an IFFT on the buffer
    vd.ifft(test_img)

    assert np.allclose(test_img.read(0), np.fft.ifftn(signal_3d) * np.prod(signal_3d.shape), atol=0.01) or np.allclose(test_img.read(0), np.fft.ifftn(signal_3d), atol=0.01)

# def test_fft_2d_384x384():
#     # Create a 2D buffer
#     signal_2d = make_random_complex_signal((384, 384))

#     test_img = vd.Buffer(signal_2d.shape, vd.complex64)
#     test_img.write(signal_2d)

#     # Perform an FFT on the buffer
#     vd.fft(test_img)

#     from matplotlib import pyplot as plt

#     plt.imshow(np.abs(test_img.read(0)))
#     plt.show()

#     print("Mean diff:", np.mean(np.abs(test_img.read(0) - np.fft.fft2(signal_2d))))

#     assert np.allclose(test_img.read(0), np.fft.fft2(signal_2d), atol=0.0001)

