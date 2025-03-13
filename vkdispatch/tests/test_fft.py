import vkdispatch as vd

import numpy as np

def make_random_complex_signal(shape):
    r = np.random.random(size=shape)
    i = np.random.random(size=shape)
    return (r + i * 1j).astype(np.complex64)

def make_square_signal(shape):
    signal = np.zeros(shape)
    signal[shape[0]//4:3*shape[0]//4, shape[1]//4:3*shape[1]//4] = 1
    return signal

def make_gaussian_signal(shape):
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    xx, yy = np.meshgrid(x, y)
    signal = np.exp(-xx**2 - yy**2)
    return signal

def test_fft_1d():
    #vd.set_log_level(vd.LogLevel.INFO)

    # Create a 1D buffer
    signal = make_random_complex_signal((50,))

    test_line = vd.Buffer((signal.shape[0],), vd.complex64)
    test_line.write(signal)

    # Perform an FFT on the buffer
    vd.fft(test_line)

    assert np.allclose(test_line.read(0), np.fft.fft(signal), atol=0.0001)

def test_fft_2d():
    # Create a 2D buffer
    signal_2d = make_random_complex_signal((50, 50))

    test_img = vd.Buffer(signal_2d.shape, vd.complex64)
    test_img.write(signal_2d)

    # Perform an FFT on the buffer
    vd.fft(test_img)

    assert np.allclose(test_img.read(0), np.fft.fft2(signal_2d), atol=0.001)

def test_fft_3d():
    # Create a 3D buffer
    signal_3d = make_random_complex_signal((50, 50, 50))

    test_img = vd.Buffer(signal_3d.shape, vd.complex64)
    test_img.write(signal_3d)

    # Perform an FFT on the buffer
    vd.fft(test_img)

    assert np.allclose(test_img.read(0), np.fft.fftn(signal_3d), atol=0.01)

def test_fft_2d_batch():
    # Create a 3D buffer
    signal_3d = make_random_complex_signal((50, 50, 50))

    test_img = vd.Buffer(signal_3d.shape, vd.complex64)
    test_img.write(signal_3d)

    # Perform an FFT on the buffer
    vd.fft(test_img, axes=[1, 2])

    assert np.allclose(test_img.read(0), np.fft.fftn(signal_3d, axes=[1, 2]), atol=0.01)

def test_ifft_1d():
    # Create a 1D buffer
    signal = make_random_complex_signal((50, ))

    test_line = vd.Buffer((signal.shape[0],), vd.complex64)
    test_line.write(signal)

    # Perform an IFFT on the buffer
    vd.ifft(test_line)
    
    assert np.allclose(test_line.read(0), np.fft.ifft(signal) * np.prod(signal.shape), atol=0.0001)

def test_ifft_2d():
    # Create a 2D buffer
    signal_2d = make_random_complex_signal((50, 50))

    test_img = vd.Buffer(signal_2d.shape, vd.complex64)
    test_img.write(signal_2d)

    output = vd.Buffer(signal_2d.shape, vd.complex64)

    # Perform an IFFT on the buffer
    vd.ifft(output, test_img)

    assert np.allclose(output.read(0), np.fft.ifft2(signal_2d) * np.prod(signal_2d.shape), atol=0.001)

def test_ifft_3d():
    # Create a 3D buffer
    signal_3d = make_random_complex_signal((50, 50, 50))

    test_img = vd.Buffer(signal_3d.shape, vd.complex64)
    test_img.write(signal_3d)

    # Perform an IFFT on the buffer
    vd.ifft(test_img)

    assert np.allclose(test_img.read(0), np.fft.ifftn(signal_3d) * np.prod(signal_3d.shape), atol=0.01)

def test_rfft_1d():
    # Create a 1D buffer
    signal = np.random.random((50,))

    test_line = vd.asrfftbuffer(signal)

    # Perform an RFFT on the buffer
    vd.rfft(test_line)

    assert np.allclose(test_line.read_fourier(0), np.fft.rfft(signal), atol=0.01)

def test_rfft_2d():
    # Create a 2D buffer
    signal_2d = np.random.random((50, 50))

    test_img = vd.asbuffer(signal_2d.astype(np.float32))
    output = vd.RFFTBuffer(signal_2d.shape)

    # Perform an RFFT on the buffer
    vd.rfft(output, test_img)

    assert np.allclose(output.read_fourier(0), np.fft.rfft2(signal_2d), atol=0.01)

def test_rfft_3d():
    # Create a 3D buffer
    signal_3d = np.random.random((50, 50, 50))

    test_img = vd.asrfftbuffer(signal_3d)

    # Perform an RFFT on the buffer
    vd.rfft(test_img)

    assert np.allclose(test_img.read_fourier(0), np.fft.rfftn(signal_3d), atol=0.01)

def test_rfft_2d_batch():
    # Create a 3D buffer
    signal_3d = np.random.random((50, 50, 50))

    test_img = vd.RFFTBuffer(signal_3d.shape)
    test_img.write_real(signal_3d)

    # Perform an FFT on the buffer
    vd.rfft(test_img, axes=[1, 2])

    assert np.allclose(test_img.read_fourier(0), np.fft.rfftn(signal_3d, axes=[1, 2]), atol=0.01)

def test_irfft_1d():
    # Create a 1D buffer
    signal = np.random.random((50,))

    signal_complex = np.fft.rfft(signal)

    test_line = vd.RFFTBuffer(signal.shape)
    test_line.write_fourier(signal_complex)

    # Perform an IRFFT on the buffer
    vd.irfft(test_line)

    assert np.allclose(test_line.read_real(0), signal * np.prod(test_line.real_shape), atol=0.01)

def test_irfft_2d():
    # Create a 2D buffer
    signal_2d = np.random.random((50, 50))

    signal_2d_complex = np.fft.rfft2(signal_2d)

    test_img = vd.RFFTBuffer(signal_2d.shape)
    test_img.write_fourier(signal_2d_complex)

    # Perform an IRFFT on the buffer
    vd.irfft(test_img)

    assert np.allclose(test_img.read_real(0), signal_2d * np.prod(test_img.real_shape), atol=0.01)

def test_irfft_3d():
    # Create a 3D buffer
    signal_3d = np.random.random((50, 50, 50))

    signal_3d_complex = np.fft.rfftn(signal_3d)

    test_img = vd.RFFTBuffer(signal_3d.shape)
    test_img.write_fourier(signal_3d_complex)

    # Perform an IRFFT on the buffer
    vd.irfft(test_img)

    calculated_value = test_img.read_real(0)
    expected_value = signal_3d * np.prod(test_img.real_shape)

    assert np.allclose(calculated_value, expected_value, atol=0.1)

def test_fft_2d_384x384():
    # Create a 2D buffer
    signal_2d = make_random_complex_signal((384, 384))

    test_img = vd.Buffer(signal_2d.shape, vd.complex64)
    test_img.write(signal_2d)

    # Perform an FFT on the buffer
    vd.fft(test_img)

    assert np.allclose(test_img.read(0), np.fft.fft2(signal_2d), atol=0.01)

def cpu_convolve_2d(signal_2d, kernel_2d):
    return np.fft.irfft2(
        (np.fft.rfft2(signal_2d).astype(np.complex64) 
        * np.fft.rfft2(kernel_2d).astype(np.complex64))
    .astype(np.complex64))

def test_convolution_2d():
    # Create a 2D buffer
   
    side_len = 50

    signal_2d = np.fft.fftshift(np.abs(make_gaussian_signal((side_len, side_len)))).astype(np.float32)
    kernel_2d = np.fft.fftshift(np.abs(make_square_signal((side_len, side_len)))).astype(np.float32).reshape((1, side_len, side_len))

    input_buffer = vd.asbuffer(signal_2d)
    test_img = vd.asrfftbuffer(signal_2d)
    kernel_img = vd.asrfftbuffer(kernel_2d)

    vd.create_kernel_2Dreal(kernel_img)

    # Perform an FFT on the buffer
    vd.convolve_2Dreal(test_img, kernel_img, input=input_buffer, normalize=True)
    
    result = test_img.read_real(0)
    reference = cpu_convolve_2d(signal_2d, kernel_2d[0])
    
    #print(result.mean())
    #print(reference.mean())
    
    #print((result - reference).mean())
    
    assert np.allclose(result, reference, atol=0.1)
