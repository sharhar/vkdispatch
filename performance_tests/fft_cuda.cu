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
    int axis;          // 0 or 1
    int iter_count;
    int iter_batch;
    int run_count;
    int warmup = 10;   // match Torch script’s warmup
};

static Config parse_args(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_size> <axis> <iter_count> <iter_batch> <run_count>\n";
        std::exit(1);
    }
    Config c;
    c.data_size  = std::stoll(argv[1]);
    c.axis       = std::stoi(argv[2]);
    c.iter_count = std::stoi(argv[3]);
    c.iter_batch = std::stoi(argv[4]);
    c.run_count  = std::stoi(argv[5]);
    if (c.axis != 0 && c.axis != 1) {
        std::cerr << "axis must be 0 or 1\n";
        std::exit(1);
    }
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
    const double bytes = 2.0 * static_cast<double>(dim0) * static_cast<double>(dim1) * 8.0;
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

static double run_cufft_case(const Config& cfg, int fft_size) {
    // Shape has two dims; size along 'axis' is fft_size, the other is data_size / fft_size
    const int batched_axis = (cfg.axis + 1) % 2;

    long long dims[2] = {0, 0};
    dims[cfg.axis] = fft_size;
    dims[batched_axis] = cfg.data_size / fft_size;

    if (dims[batched_axis] <= 0) {
        // Nothing to do (mismatch), return 0
        return 0.0;
    }

    const long long dim0 = dims[0];
    const long long dim1 = dims[1];
    const long long total_elems = dim0 * dim1;

    // Device buffers (in-place transform will overwrite input)
    cufftComplex* d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, total_elems * sizeof(cufftComplex)), "cudaMalloc d_data");
    // Optionally zero-fill
    checkCuda(cudaMemset(d_data, 0, total_elems * sizeof(cufftComplex)), "cudaMemset d_data");

    // cuFFT plan: 1D transforms along 'axis', batched over the other dimension
    cufftHandle plan;
    checkCuFFT(cufftCreate(&plan), "cufftCreate");

    int n[1] = { fft_size }; // 1D length

    // Strides and distances depend on axis in row-major layout:
    // Layout: index (i0, i1) -> offset = i0*dim1 + i1
    int istride, ostride, idist, odist, batch;
    int inembed[1] = {0};  // not used when rank=1 unless advanced layouts; set 0
    int onembed[1] = {0};

    if (cfg.axis == 1) {
        // Transform along contiguous last dimension (dim1)
        // Each transform is a row of length dim1; there are dim0 batches.
        istride = 1;
        ostride = 1;
        idist   = static_cast<int>(dim1);
        odist   = static_cast<int>(dim1);
        batch   = static_cast<int>(dim0);
    } else {
        // Transform along first dimension (dim0)
        // Elements of a single transform are separated by dim1
        istride = static_cast<int>(dim1);
        ostride = static_cast<int>(dim1);
        idist   = 1;
        odist   = 1;
        batch   = static_cast<int>(dim1);
    }

    // Create plan (complex-to-complex, single-precision)
    checkCuFFT(
        cufftPlanMany(&plan,
                      /*rank*/ 1,
                      n,
                      inembed,  istride, idist,
                      onembed,  ostride, odist,
                      CUFFT_C2C,
                      batch),
        "cufftPlanMany"
    );

    // Warmup
    for (int i = 0; i < cfg.warmup; ++i) {
        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "cufftExecC2C warmup");
    }
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (after warmup)");

    // Time iter_count * iter_batch executions
    cudaEvent_t ev_start, ev_stop;
    checkCuda(cudaEventCreate(&ev_start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&ev_stop),  "cudaEventCreate stop");
    checkCuda(cudaEventRecord(ev_start),  "cudaEventRecord start");

    for (int it = 0; it < cfg.iter_count; ++it) {
        checkCuFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "cufftExecC2C timed");
    }

    checkCuda(cudaEventRecord(ev_stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(ev_stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, ev_start, ev_stop), "cudaEventElapsedTime");

    // Cleanup events (keep data around until after sync)
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // Convert elapsed to seconds
    const double seconds = static_cast<double>(ms) / 1000.0;

    // Compute throughput in GB/s (same accounting as Torch: 2 * elems * 8 bytes per exec)
    const double gb_per_exec_once = gb_per_exec(dim0, dim1);
    const double total_execs = static_cast<double>(cfg.iter_count); // * static_cast<double>(cfg.iter_batch);
    const double gb_per_second = (total_execs * gb_per_exec_once) / seconds;

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);

    return gb_per_second;
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    const auto sizes = get_fft_sizes();

    const std::string output_name = "fft_cuda_" + std::to_string(cfg.axis) + "_axis.csv";
    std::ofstream out(output_name);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_name << "\n";
        return 1;
    }

    std::cout << "Running cuFFT tests with data size " << cfg.data_size
              << ", axis " << cfg.axis
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
        out << "cuda," << fft_size;
        out << std::fixed << std::setprecision(2);
        for (double v : rates) out << "," << v;
        out << "," << mean << "," << stdev << "\n";
    }

    std::cout << "Results saved to " << output_name << "\n";
    return 0;
}
