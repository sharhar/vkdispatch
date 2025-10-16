// actual_test_cuda.cu
// Usage: ./actual_test_cuda <data_size> <axis> <iter_count> <iter_batch> <run_count>
// Output: fft_cuda_<axis>_axis.csv with the same columns as your Torch script.
//
// Build (example):
//   nvcc -O3 -std=c++17 actual_test_cuda.cu -lcufft -o actual_test_cuda

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

__global__ void fill_randomish(cufftComplex* a, long long n){
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if(i<n){
        float x = __sinf(i * 0.00173f);
        float y = __cosf(i * 0.00091f);
        a[i] = make_float2(x, y);
    }
}

__global__ void scale_kernel(cufftComplex* data, float scale_factor, long long total_elems) {
    long long i = blockIdx.x * 1LL * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        data[i].x *= scale_factor;
        data[i].y *= scale_factor;
    }
}

static inline void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] " << what << " failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

static inline void checkCuFFT(cufftResult err, const char* what) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << "[cuFFT] " << what << " failed: " << err << "\n";
        std::exit(1);
    }
}

struct Config {
    long long data_size;
    int iter_count;
    int iter_batch;
    int run_count;
    int warmup = 10;   // match Torch script’s warmup
};

static Config parse_args(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_size> <iter_count> <iter_batch> <run_count>\n";
        std::exit(1);
    }
    Config c;
    c.data_size  = std::stoll(argv[1]);
    c.iter_count = std::stoi(argv[2]);
    c.iter_batch = std::stoi(argv[3]);
    c.run_count  = std::stoi(argv[4]);
    return c;
}

static std::vector<int> get_fft_sizes() {
    std::vector<int> sizes;
    for (int p = 6; p <= 12; ++p) sizes.push_back(1 << p); // 64..4096
    return sizes;
}

// Compute GB processed per single FFT execution (read + write) for shape (dim0, dim1)
static double gb_per_exec(long long dim0, long long dim1) {
    // complex64 = 8 bytes; count both read and write -> *2
    const double bytes = static_cast<double>(dim0) * static_cast<double>(dim1) * 8.0;
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

static double run_cufft_case(const Config& cfg, int fft_size) {
    const long long dim0 = cfg.data_size / fft_size;
    const long long dim1 = fft_size;
    const long long total_elems = dim0 * dim1;

    // Device buffers (in-place transform will overwrite input)
    cufftComplex* d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, total_elems * sizeof(cufftComplex)), "cudaMalloc d_data");
    // Optionally zero-fill
    checkCuda(cudaMemset(d_data, 0, total_elems * sizeof(cufftComplex)), "cudaMemset d_data");

    cufftComplex* d_kernel = nullptr;
    checkCuda(cudaMalloc(&d_kernel, (total_elems) * sizeof(cufftComplex)), "cudaMalloc d_kernel");
    // Optionally zero-fill
    checkCuda(cudaMemset(d_kernel, 0, (total_elems) * sizeof(cufftComplex)), "cudaMemset d_kernel");

    {
        int t = 256, b = int((total_elems + t - 1) / t);
        fill_randomish<<<b,t>>>(d_data, total_elems);
        checkCuda(cudaGetLastError(), "fill launch");
        checkCuda(cudaDeviceSynchronize(), "fill sync");

        int kt = 256, kb = int((total_elems + kt - 1) / kt);
        fill_randomish<<<kb,kt>>>(d_kernel, total_elems);
        checkCuda(cudaGetLastError(), "fill kernel launch");
        checkCuda(cudaDeviceSynchronize(), "fill kernel sync");
    }

    // --- plan bound to the stream ---
    cufftHandle plan;
    checkCuFFT(cufftCreate(&plan), "cufftCreate");

    // int n[2] = { int(dim1), int(dim2) };
    // int inembed[2] = { int(dim1), int(dim2) };        // physical layout (same as n for tight pack)
    // int onembed[2] = { int(dim1), int(dim2) };
    // int istride    = 1;               // contiguous within each 2D image
    // int ostride    = 1;
    // int idist      = int(dim1)* int(dim2);           // distance between images
    // int odist      = int(dim1)* int(dim2);

    // checkCuFFT(cufftPlanMany(&plan, 2, n,
    //                               inembed,  istride, idist,
    //                               onembed,  ostride, odist,
    //                               CUFFT_C2C, int(dim0)), "plan2d");

    checkCuFFT(cufftPlan1d(&plan, dim1, CUFFT_C2C, dim0), "plan");

    // --- warmup on the stream ---
    for (int i = 0; i < cfg.warmup; ++i) {
        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "warmup");
        scale_kernel<<<(total_elems+255)/256,256>>>(d_data, 5.0, total_elems);
        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE), "warmup");
    }
    
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    // === OPTION A: plain single-stream timing (simple & robust) ===
    cudaEvent_t evA, evB;
    checkCuda(cudaEventCreate(&evA), "evA");
    checkCuda(cudaEventCreate(&evB), "evB");
    checkCuda(cudaEventRecord(evA), "record A");
    for (int it = 0; it < cfg.iter_count; ++it) {
        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "exec");
        scale_kernel<<<(total_elems+255)/256,256>>>(d_data, 5.0, total_elems);
        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE), "exec");
    }
    checkCuda(cudaEventRecord(evB), "record B");
    checkCuda(cudaEventSynchronize(evB), "sync B");
    checkCuda(cudaDeviceSynchronize(), "warmup sync");
    float ms = 0.f; checkCuda(cudaEventElapsedTime(&ms, evA, evB), "elapsed");
    checkCuda(cudaEventDestroy(evA), "dA");
    checkCuda(cudaEventDestroy(evB), "dB");

    // Convert elapsed to seconds
    const double seconds = static_cast<double>(ms) / 1000.0;

    // Compute throughput in GB/s (same accounting as Torch: 2 * elems * 8 bytes per exec)
    const double gb_per_exec_once = 6 * gb_per_exec(dim0, dim1);
    const double total_execs = static_cast<double>(cfg.iter_count); // * static_cast<double>(cfg.iter_batch);
    const double gb_per_second = (total_execs * gb_per_exec_once) / seconds;

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    cudaFree(d_kernel);

    return gb_per_second;
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    const auto sizes = get_fft_sizes();

    const std::string output_name = "conv_nonstrided_cufft.csv";
    std::ofstream out(output_name);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_name << "\n";
        return 1;
    }

    std::cout << "Running cuFFT tests with data size " << cfg.data_size
              << ", iter_count " << cfg.iter_count
              << ", iter_batch " << cfg.iter_batch
              << ", run_count " << cfg.run_count << "\n";

    // Header: Backend, FFT Size, Run 1..N, Mean, Std Dev
    out << "Backend,FFT Size";
    for (int i = 0; i < cfg.run_count; ++i) out << ",Run " << (i + 1) << " (GB/s)";
    out << ",Mean,Std Dev\n";

    for (int fft_size : sizes) {
        std::vector<double> rates;
        rates.reserve(cfg.run_count);

        for (int r = 0; r < cfg.run_count; ++r) {
            const double gbps = run_cufft_case(cfg, fft_size);
            std::cout << "FFT Size: " << fft_size << ", Throughput: " << std::fixed << std::setprecision(2)
                      << gbps << " GB/s\n";
            rates.push_back(gbps);
        }

        // Compute mean/std
        double mean = 0.0;
        for (double v : rates) mean += v;
        mean /= static_cast<double>(rates.size());

        double var = 0.0;
        for (double v : rates) {
            const double d = v - mean;
            var += d * d;
        }
        var /= static_cast<double>(rates.size());
        const double stdev = std::sqrt(var);

        // Round to 2 decimals like your Torch script
        out << "cufft," << fft_size;
        out << std::fixed << std::setprecision(2);
        for (double v : rates) out << "," << v;
        out << "," << mean << "," << stdev << "\n";
    }

    std::cout << "Results saved to " << output_name << "\n";
    return 0;
}
