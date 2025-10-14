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

# /usr/local/cuda/bin/nvcc -O2 -std=c++17 ../conv_cufft.cu -rdc=true -lcufft_static -lculibos -o conv_cufft.exec
# /usr/local/cuda/bin/nvcc -O2 -std=c++17 ../conv_cufft_callback.cu -rdc=true -lcufft_static -lculibos -o conv_cufft_callback.exec

echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"

# echo "Running cuFFT FFT..."
# ./conv_cufft.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running cuFFT with callbacks FFT..."
# ./conv_cufft_callback.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running VKFFT FFT..."
# python3 ../conv_vkfft.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running Vkdispatch FFT..."
# python3 ../conv_vkdispatch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running Vkdispatch None FFT..."
# python3 ../conv_vkdispatch_none.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running Vkdispatch Sdata FFT..."
# python3 ../conv_vkdispatch_sdata.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running Vkdispatch Compute FFT..."
python3 ../conv_vkdispatch_compute.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running PyTorch FFT..."
# python3 ../conv_torch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

# echo "Running ZipFFT FFT..."
# python3 ../conv_zipfft.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

python3 ../conv_make_graph.py