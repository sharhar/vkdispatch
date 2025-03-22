import numpy as np

def fft_decomposition(signal, M, N):
    """
    Compute a length M*N FFT using the row-column decomposition method.
    
    This function computes the DFT of a signal by:
    1. Rearranging the signal into a M×N matrix
    2. Computing N DFTs of length M
    3. Applying twiddle factors
    4. Computing M DFTs of length N
    5. Rearranging the result
    
    Args:
        signal: Input signal of length M*N
        M: First dimension size
        N: Second dimension size
        
    Returns:
        The DFT of the input signal (length M*N)
    """
    assert len(signal) == M * N, f"Signal must be of length {M*N}"
    
    # Step 1: Reshape the signal into a M×N matrix (column-major order)
    signal_matrix = np.reshape(signal, (M, N), order='F')
    
    # Step 2: Compute N FFTs of length M (along columns)
    intermediate = np.zeros((M, N), dtype=complex)
    for k in range(N):
        intermediate[:, k] = np.fft.fft(signal_matrix[:, k])
    
    # Step 3: Apply twiddle factors
    for m in range(M):
        for k in range(N):
            intermediate[m, k] *= np.exp(-2j * np.pi * m * k / (M * N))
    
    # Step 4: Compute M FFTs of length N (along rows)
    result_matrix = np.zeros((M, N), dtype=complex)
    for m in range(M):
        result_matrix[m, :] = np.fft.fft(intermediate[m, :])
    
    # Step 5: Rearrange the result to get the final DFT
    result = np.zeros(M * N, dtype=complex)
    for r in range(M):
        for q in range(N):
            p = q * M + r  # Row-column to linear index mapping
            result[p] = result_matrix[r, q]
    
    return result

def verify_fft_decomposition(signal, M, N):
    """
    Verify the correctness of the FFT decomposition by comparing with numpy's FFT.
    
    Args:
        signal: Input signal of length M*N
        M: First dimension size
        N: Second dimension size
        
    Returns:
        Maximum absolute error between the two methods
    """
    # Compute FFT using our decomposition method
    fft_decomp = fft_decomposition(signal, M, N)
    
    # Compute FFT using numpy's built-in FFT
    fft_numpy = np.fft.fft(signal)
    
    # Print maximum absolute error
    max_error = np.max(np.abs(fft_decomp - fft_numpy))
    print(f"Maximum absolute error: {max_error}")
    
    return max_error

# Example usage for a 15-point FFT (M=3, N=5)
def example_length_15():
    # Generate a random signal of length 15
    signal = np.random.rand(15) + 1j * np.random.rand(15)
    
    # Define dimensions
    M, N = 3, 5
    
    # Verify correctness
    verify_fft_decomposition(signal, M, N)
    
    # Compute FFT using our decomposition
    result = fft_decomposition(signal, M, N)
    
    # Return the result
    return result

# For larger examples (e.g., M=4, N=8 for a 32-point FFT)
def example_length_32():
    # Generate a random signal of length 32
    signal = np.random.rand(32) + 1j * np.random.rand(32)
    
    # Define dimensions
    M, N = 4, 8
    
    # Verify correctness
    verify_fft_decomposition(signal, M, N)
    
    # Compute FFT using our decomposition
    result = fft_decomposition(signal, M, N)
    
    # Return the result
    return result

if __name__ == "__main__":
    print("Testing 15-point FFT decomposition:")
    example_length_15()
    
    print("\nTesting 32-point FFT decomposition:")
    example_length_32()