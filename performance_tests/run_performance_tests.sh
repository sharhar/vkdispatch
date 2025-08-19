#!/bin/bash

mkdir -p test_results

cd test_results

python3 ../kernels_per_streams.py 10 1 # Test with up to 10 streams and 1 device
python3 ../kernels_per_streams.py 10 2 # Test with up to 10 streams and 2 devices
python3 ../kernels_per_streams.py 10 3 # Test with up to 10 streams and 3 devices
python3 ../kernels_per_streams.py 10 4 # Test with up to 10 streams and 4 devices

python3 ../kernels_per_batch_size.py 1 1 # Test batch sizes with 1 device and 1 stream
python3 ../kernels_per_batch_size.py 2 1 # Test batch sizes with 1 device and 2 streams
python3 ../kernels_per_batch_size.py 4 1 # Test batch sizes with 1 device and 4 streams

python3 ../kernels_per_batch_size.py 1 4 # Test batch sizes with 4 device and 1 stream
python3 ../kernels_per_batch_size.py 2 4 # Test batch sizes with 4 device and 2 streams
python3 ../kernels_per_batch_size.py 4 4 # Test batch sizes with 4 device and 3 streams