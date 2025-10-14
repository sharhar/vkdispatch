#!/bin/bash

mkdir -p test_results

cd test_results
#DATA_SIZE=134217728
DATA_SIZE=67108864
#DATA_SIZE=33554432
SIGNAL_FACTOR=8
ITER_COUNT=80
BATCH_SIZE=10
REPEATS=3

# /usr/local/cuda/bin/nvcc ../fft_cufft.cu -o fft_cufft.exec -lcufft

echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"

#echo "Running cuFFT FFT..."
#./fft_cufft.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running Vkdispatch FFT..."
# python3 ../fft_strided_vkdispatch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running VKFFT FFT..."
# python3 ../fft_strided_vkfft.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running PyTorch FFT..."
python3 ../fft_strided_torch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running ZipFFT FFT..."
# python3 ../fft_strided_zipfft.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running ZipFFT NO Compute FFT..."
# python3 ../fft_strided_zipfft_no_compute.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

python3 ../fft_strided_make_graph.py