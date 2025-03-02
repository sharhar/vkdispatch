import numpy as np
import matplotlib.pyplot as plt

a = eval("""lambda a, b: a + b""")

print(a)

exit()

# Create a sample non-square signal (e.g., 300x500)
H, W = 301, 501  # Odd dimensions
x = np.linspace(-5, 5, W)
y = np.linspace(-5, 5, H)
X, Y = np.meshgrid(x, y)
signal = np.exp(-X**2 - Y**2)  # Gaussian function

# Compute its Fourier transform
F_signal = np.fft.fft2(signal)

# Generate the phase shift factor
kx = np.fft.fftfreq(W) * W  # Frequency indices for width
ky = np.fft.fftfreq(H) * H  # Frequency indices for height
KX, KY = np.meshgrid(kx, ky)

phase_shift = np.exp(1j * np.pi * (KX + KY))  # Phase factor

# Apply the phase shift in Fourier space
F_shifted = F_signal * phase_shift

# Inverse Fourier Transform back to real space
shifted_signal = np.fft.ifft2(F_shifted).real  # Take real part

# Compare original and shifted results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(signal, cmap='gray')
ax[0].set_title("Original Signal")

ax[1].imshow(shifted_signal, cmap='gray')
ax[1].set_title("Shifted Signal (FFT-based)")

plt.show()
