#!/bin/bash

mkdir -p test_results

cd test_results

python3 ../kernels_per_streams.py 10 1 # Test with up to 10 streams and 1 device
python3 ../kernels_per_streams.py 10 2 # Test with up to 10 streams and 2 devices
python3 ../kernels_per_streams.py 10 3 # Test with up to 10 streams and 3 devices
python3 ../kernels_per_streams.py 10 4 # Test with up to 10 streams and 4 devices

