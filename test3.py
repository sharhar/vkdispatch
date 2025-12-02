def get_cuda_device_map():
    """
    Returns a dict mapping CUDA device index -> UUID (bytes).
    Format: { 0: b'\x00...', 1: b'\x01...' }
    """
    try:
        from cuda.bindings import driver
    except ImportError as e:
        # If the cuda driver bindings are not available, just return None
        return None

    # 1. Initialize the CUDA Driver API
    err, = driver.cuInit(0)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Failed to initialize CUDA Driver API")

    # 2. Get device count
    err, count = driver.cuDeviceGetCount()
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Failed to get CUDA device count")

    uuid_map = {}

    # 3. Iterate through devices and fetch UUIDs
    for i in range(count):
        # Get handle for device i
        err, device = driver.cuDeviceGet(i)
        if err != driver.CUresult.CUDA_SUCCESS:
            continue

        # Get UUID (returns tuple: (error, bytes))
        err, uuid_bytes = driver.cuDeviceGetUuid(device)
        if err == driver.CUresult.CUDA_SUCCESS:
            # uuid_bytes is already a 16-byte object, matches Vulkan format
            uuid_map[i] = uuid_bytes.bytes

    return uuid_map

# Example usage to print them out
if __name__ == "__main__":
    try:
        device_map = get_cuda_device_map()
        for idx, uuid in device_map.items():
            # Convert bytes to hex string for readability (e.g., "54a...e12")
            print(f"CUDA Device {idx}: UUID={uuid.hex()}")

            uuid_str = '-'.join([
                uuid[0:4].hex(),
                uuid[4:6].hex(),
                uuid[6:8].hex(),
                uuid[8:10].hex(),
                uuid[10:16].hex(),
            ])
            print(f"\tUUID: {uuid_str}")
    except Exception as e:
        print(f"Error: {e}")