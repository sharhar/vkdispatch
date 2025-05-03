import numpy as np
from typing import List

import vkdispatch as vd

def default_register_limit():
    if "NVIDIA" in vd.get_context().device_infos[0].device_name:
        return 16

    return 15

def prime_factors(n) -> List[int]:
    factors = []
    
    # Handle the factor 2 separately
    while n % 2 == 0:
        factors.append(2)
        n //= 2
        
    # Now handle odd factors
    factor = 3
    while factor * factor <= n:
        while n % factor == 0:
            factors.append(factor)
            n //= factor
        factor += 2
        
    # If at the end, n is greater than 1, it is a prime number itself
    if n > 1:
        factors.append(n)
        
    return factors

def group_primes(primes, register_count):
    groups: List[List] = []

    for prime in primes:
        if len(groups) == 0:
            groups.append([prime])
            continue

        if np.prod(groups[-1]) * prime <= register_count:
            groups[-1].append(prime)
            continue

        groups.append([prime])

    return groups

def pad_dim(dim: int, max_register_count: int = default_register_limit()):
    assert dim > 0, 'Dimension must be greater than 0'

    current_dim = dim
    current_primes = prime_factors(current_dim)

    while any([prime > max_register_count for prime in current_primes]):
        current_dim += 1
        current_primes = prime_factors(current_dim)

    return current_dim