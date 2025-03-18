import numpy as np

def cooley_tukey_fft(x):
    """
    Compute the FFT of x using the Cooley-Tukey algorithm.
    x should have a length that is a power of 2.
    """
    N = len(x)
    
    # Base case: FFT of size 1 is just the input
    if N == 1:
        return x
    
    # Check if N is a power of 2
    if N & (N-1) != 0:
        raise ValueError("Input length must be a power of 2")
    
    # Split the input into even and odd indices
    even = cooley_tukey_fft(x[0::2])
    odd = cooley_tukey_fft(x[1::2])
    
    # Combine the results
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    first_half = even + factor[:N//2] * odd
    second_half = even + factor[N//2:] * odd
    
    return np.concatenate([first_half, second_half])

def factored_fft(x, A, B):
    """
    Compute FFT of x by factoring into A FFTs of size B and B FFTs of size A.
    N = A * B must be the length of x.
    """
    N = len(x)
    if N != A * B:
        raise ValueError(f"Input length {N} must equal A*B = {A*B}")
    
    # Reshape input to A rows and B columns
    x_reshaped = np.array(x).reshape(A, B)
    
    # Step 1: Compute B FFTs of length A
    rows_fft = np.zeros((A, B), dtype=complex)
    for b in range(B):
        rows_fft[:, b] = np.fft.fft(x_reshaped[:, b])
    
    # Step 2: Apply twiddle factors
    for a in range(A):
        for b in range(B):
            rows_fft[a, b] *= np.exp(-2j * np.pi * a * b / N)
    
    # Step 3: Compute A FFTs of length B
    result = np.zeros((A, B), dtype=complex)
    for a in range(A):
        result[a, :] = np.fft.fft(rows_fft[a, :])
    
    # Step 4: Transpose and reshape to get the final result
    return result.T.reshape(N)

# Example usage
def test_fft():
    # Test with a simple signal
    N = 16
    t = np.linspace(0, 1, N, endpoint=False)
    x = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 4 * t)
    
    # Compute FFT using numpy's implementation
    fft_numpy = np.fft.fft(x)
    
    # Compute FFT using our Cooley-Tukey implementation
    fft_ct = cooley_tukey_fft(x)
    
    # Compute FFT using our factored implementation
    A, B = 4, 4  # 4*4 = 16
    fft_factored = factored_fft(x, A, B)
    
    # Compare results
    print("Numpy FFT:", fft_numpy)
    print("Cooley-Tukey FFT:", fft_ct)
    print("Factored FFT:", fft_factored)
    
    print("\nMax absolute difference between numpy and Cooley-Tukey:", 
          np.max(np.abs(fft_numpy - fft_ct)))
    print("Max absolute difference between numpy and factored:", 
          np.max(np.abs(fft_numpy - fft_factored)))

if __name__ == "__main__":
    test_fft()