import math

def calculate_registers_per_thread(fft_size, max_threads=1024, aim_threads=256, 
                                   warp_size=32, register_boost=1, vendor_id=0x10DE,
                                   axis_id=0, num_uploads=1, grouped_batch=1):
    """
    Calculate optimal registers per thread for FFT scheduling.
    
    vendor_id: 0x10DE (NVIDIA), 0x1002 (AMD)
    """
    
    # Factor the FFT size into prime radices
    radices = factorize(fft_size, max_radix=7)  # [2, 2, 2, 3, 5, ...] etc
    
    # Try different stage decompositions (1 to max possible)
    max_stages = len(radices)
    best_config = None
    best_score = -1e9
    
    for num_stages in range(1, max_stages + 1):
        # Get all possible ways to group radices into num_stages
        stage_splits = find_stage_splits(radices, num_stages)
        
        for split in stage_splits:
            # split is like [8, 4, 16] meaning radices [2,2,2], [2,2], [2,2,2,2]
            config = evaluate_split(split, fft_size, max_threads, aim_threads,
                                   warp_size, register_boost, vendor_id, 
                                   axis_id, num_uploads, grouped_batch)
            
            if config['score'] > best_score:
                best_score = config['score']
                best_config = config
    
    return best_config['registers_per_thread']


def evaluate_split(split, fft_size, max_threads, aim_threads, warp_size, 
                   register_boost, vendor_id, axis_id, num_uploads, grouped_batch):
    """
    Evaluate a particular stage decomposition.
    split: list of radices for each stage, e.g., [8, 16, 8] for 1024-point FFT
    """
    
    # For each stage, calculate threads needed
    threads_per_stage = [math.ceil(fft_size / radix) for radix in split]
    min_threads = min(threads_per_stage)
    max_threads_needed = max(threads_per_stage)
    
    # Try different actual thread counts
    max_range = min(max_threads * register_boost, max_threads_needed)
    best_score = -1e9
    best_regs = {}
    
    for actual_threads in range(1, max_range + 1):
        # Skip redundant thread counts (optimization)
        effective_threads = {}
        skip = False
        
        for i, (radix, threads_needed) in enumerate(zip(split, threads_per_stage)):
            if threads_needed > actual_threads:
                # Need multiple batches per thread
                effective = math.ceil(threads_needed / 
                                     math.ceil(threads_needed / actual_threads))
            else:
                effective = threads_needed
            effective_threads[i] = effective
        
        # All stages must fit in max_threads
        max_effective = max(effective_threads.values())
        if max_effective > max_threads * register_boost:
            continue
            
        # Calculate registers per stage
        registers_per_stage = {}
        for i, (radix, threads_needed) in enumerate(zip(split, threads_per_stage)):
            registers_per_stage[i] = radix * math.ceil(threads_needed / max_effective)
        
        min_regs = min(registers_per_stage.values())
        max_regs = max(registers_per_stage.values())
        
        # Calculate score
        score = 0
        
        # Penalty for register imbalance
        if min_regs > 0:
            imbalance = (max_regs / min_regs - 1) ** 2
            score -= imbalance * 0.001
        
        # Penalty for too many stages
        score -= 0.002 * len(split)
        
        # Penalty for high register count
        register_threshold = get_register_threshold(vendor_id, fft_size)
        score -= 0.00005 * min(max_regs, register_threshold)
        if max_regs > register_threshold:
            score -= 0.001 * (max_regs - register_threshold)
        
        # Penalty for poor warp alignment
        refine_batch = grouped_batch
        if axis_id == 0 and num_uploads == 1:
            if max_effective < aim_threads:
                refine_batch = aim_threads // max_effective
                if refine_batch == 0:
                    refine_batch = 1
            else:
                refine_batch = 1
        
        if vendor_id == 0x10DE:  # NVIDIA prefers power-of-2
            refine_batch = 2 ** math.ceil(math.log2(refine_batch))
        
        total_threads = refine_batch * max_effective
        if total_threads % warp_size != 0:
            warp_efficiency = (total_threads % warp_size) / warp_size
            score -= (1.0 - warp_efficiency) * 0.001
        
        # Bonus for good configurations
        if fft_size % min_regs == 0:
            if axis_id == 0 and num_uploads == 1:
                num_min_stages = sum(1 for r in registers_per_stage.values() 
                                    if r == min_regs)
                if refine_batch == 1:
                    score += 0.002 * min(num_min_stages, 2)
                elif refine_batch > 1:
                    score += 0.004
        
        if score > best_score:
            best_score = score
            best_regs = {
                'registers_per_thread': max_regs,
                'min_registers_per_thread': min_regs,
                'registers_per_radix': {radix: registers_per_stage[i] 
                                       for i, radix in enumerate(split)}
            }
    
    return {'score': best_score, **best_regs}


def get_register_threshold(vendor_id, fft_size):
    """Hardware-specific register thresholds."""
    if vendor_id == 0x10DE:  # NVIDIA
        return 24 if fft_size >= 128 else 16
    else:  # AMD
        return 12


def factorize(n, max_radix=7):
    """Factor n into list of small primes up to max_radix."""
    factors = []
    for p in range(2, max_radix + 1):
        while n % p == 0:
            factors.append(p)
            n //= p
    return factors


def find_stage_splits(radices, num_stages):
    """
    Generate all ways to partition radices into num_stages groups.
    Returns product of each group, e.g., [2,2,2] -> [8]
    """
    # Simplified: just return one reasonable split
    # Full version would try all partitions
    total = 1
    for r in radices:
        total *= r
    
    if num_stages == 1:
        return [[total]]
    
    # Heuristic: try to balance stages
    splits = []
    # ... recursive partitioning logic ...
    # For simplicity, return a geometric split
    stage_size = total ** (1.0 / num_stages)
    result = []
    remaining = total
    for i in range(num_stages - 1):
        s = find_closest_factor(remaining, stage_size)
        result.append(s)
        remaining //= s
    result.append(remaining)
    
    return [result]


def find_closest_factor(n, target):
    """Find factor of n closest to target."""
    best = n
    best_diff = abs(n - target)
    for i in range(int(target), 0, -1):
        if n % i == 0:
            if abs(i - target) < best_diff:
                best = i
                best_diff = abs(i - target)
            break
    return best


# Example usage
if __name__ == "__main__":
    fft_size = 1024
    regs = calculate_registers_per_thread(fft_size,
                                          axis_id=0,
                                          max_threads=1024,
                                          aim_threads=256,
                                          warp_size=32,
                                          vendor_id=0x10DE)
    print(f"FFT size {fft_size}: {regs} registers per thread")