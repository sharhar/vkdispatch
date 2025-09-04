#!/bin/bash

mkdir -p test_results

cd test_results

DATA_SIZE=134217728
#DATA_SIZE=33554432 #134217728
SIGNAL_FACTOR=8
ITER_COUNT=200
BATCH_SIZE=10
REPEATS=5

/usr/local/cuda/bin/nvcc -O2 -std=c++17 ../conv_padded_cufft.cu -rdc=true -lcufft_static -lculibos -o conv_padded_cufft.exec
/usr/local/cuda/bin/nvcc -O2 -std=c++17 ../conv_padded_cufft_callback.cu -rdc=true -lcufft_static -lculibos -o conv_padded_cufft_callback.exec

echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Signal Factor: $SIGNAL_FACTOR"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"

echo "Running Vkdispatch FFT..."
python3 ../conv_padded_vkdispatch.py $DATA_SIZE $SIGNAL_FACTOR $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running cuFFT FFT..."
./conv_padded_cufft.exec $DATA_SIZE $SIGNAL_FACTOR $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running cuFFT callback FFT..."
./conv_padded_cufft_callback.exec $DATA_SIZE $SIGNAL_FACTOR $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running PyTorch FFT..."
python3 ../conv_padded_torch.py $DATA_SIZE $SIGNAL_FACTOR $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running ZipFFT FFT..."
python3 ../conv_padded_zipfft.py $DATA_SIZE $SIGNAL_FACTOR $ITER_COUNT $BATCH_SIZE $REPEATS

python3 ../conv_padded_make_graph.py