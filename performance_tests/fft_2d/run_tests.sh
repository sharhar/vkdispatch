#!/bin/bash

mkdir -p test_results

cd test_results

DATA_SIZE=134217728
ITER_COUNT=250
BATCH_SIZE=10
REPEATS=5

/usr/local/cuda/bin/nvcc ../fft_cuda.cu -o fft_cuda.exec -lcufft

echo "Running performance tests with the following parameters:"
echo "Data Size: $DATA_SIZE"
echo "Iteration Count: $ITER_COUNT"
echo "Batch Size: $BATCH_SIZE"
echo "Repeats: $REPEATS"

echo "Running cuFFT FFT..."
./fft_cuda.exec $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running Vkdispatch FFT..."
python3 ../fft_vkdispatch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running VKFFT FFT..."
python3 ../fft_vkfft.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

echo "Running PyTorch FFT..."
python3 ../fft_torch.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

#echo "Running ZipFFT FFT..."
#python3 ../fft_zipfft.py $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS

python3 ../fft_make_graph.py