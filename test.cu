#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call;   \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);   \
    }                         \
} while (0)

#define CHECK_CUFFT(call) do { \
    cufftResult err = call;    \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE);    \
    }                          \
} while (0)

int main() {
    // FFT parameters
    const int NX = 4096;
    const int NY = 4096;
    const int BATCH = 10;         // Number of images in batch (adjust as desired)
    const int NUM_ITER = 1000;    // Number of forward+inverse pairs

    size_t total_elems = size_t(NX) * NY * BATCH;

    // Host memory for initialization
    cufftComplex *h_input = (cufftComplex*)malloc(total_elems * sizeof(cufftComplex));
    for (size_t i = 0; i < total_elems; ++i) {
        h_input[i].x = (float)rand() / RAND_MAX;
        h_input[i].y = (float)rand() / RAND_MAX;
    }

    // Device memory
    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, total_elems * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_data, h_input, total_elems * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // cuFFT plan (batched 2D C2C)
    cufftHandle plan;
    int n[2] = {NX, NY};
    int inembed[2] = {NX, NY};
    int onembed[2] = {NX, NY};
    int istride = 1, ostride = 1;
    int idist = NX * NY, odist = NX * NY;
    CHECK_CUFFT(cufftPlanMany(&plan, 2, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, BATCH));

    // Warmup
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
    }

    CHECK_CUDA(cudaMemcpy(h_input, d_data, total_elems * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Report
    double elapsed = std::chrono::duration<double>(end - start).count();
    double pairs_per_sec = (double)NUM_ITER * BATCH / elapsed;
    printf("%d x %d x %d C2C 2D FFTs: %.2f forward+inverse pairs per second (total: %d cycles in %.3f s)\n",
           NX, NY, BATCH, pairs_per_sec, NUM_ITER, elapsed);

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_input);
    return 0;
}
