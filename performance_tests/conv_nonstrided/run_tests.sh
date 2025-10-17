#!/bin/bash

mkdir -p test_results

cd test_results

DATA_SIZE=134217728
#DATA_SIZE=67108864
#DATA_SIZE=33554432
ITER_COUNT=200
BATCH_SIZE=20
REPEATS=5

/usr/local/cuda-12.0/bin/nvcc -O2 -std=c++17 ../conv_nonstrided_cufft.cu -gencode arch=compute_86,code=sm_86 -rdc=true -lcufft_static -lculibos -o conv_nonstrided_cufft.exec

echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"

echo "Running cuFFT FFT..."
./conv_nonstrided_cufft.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running Vkdispatch FFT..."
python3 ../conv_nonstrided_vkdispatch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running PyTorch FFT..."
python3 ../conv_nonstrided_torch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running ZipFFT FFT..."
python3 ../conv_nonstrided_zipfft.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

python3 ../conv_nonstrided_make_graph.py
python3 ../conv_nonstrided_make_ratios_graph.py
