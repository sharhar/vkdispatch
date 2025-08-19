# examples/first_buffer.py
import vkdispatch
import numpy as np

def main():
    # Initialize the vkdispatch context (important!)
    vkdispatch.initialize()

    # Create a simple numpy array
    cpu_data = np.arange(16, dtype=np.int32)
    print(f"Original CPU data: {cpu_data}")

    try:
        # Create a GPU buffer
        # The 'usage' parameter is crucial for how the GPU allocates memory
        gpu_buffer = vkdispatch.BufferBuilder() \
            .size(cpu_data.nbytes) \
            .usage(vkdispatch.BufferUsage.STORAGE) \
            .build()

        # Upload data from CPU to GPU
        gpu_buffer.upload(cpu_data)
        print(f"Data uploaded to GPU: {cpu_buffer.download().flatten()}")

        # Download data back from GPU to CPU to verify
        downloaded_data = gpu_buffer.download()
        print(f"Data downloaded from GPU: {downloaded_data.flatten()}")

        # Perform any operations here (e.g., a compute shader)

        # Ensure all operations are complete and check for errors
        vkdispatch.check_for_errors()
        print("Operations completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Always deinitialize the context when done
        vkdispatch.get_context().deinitialize()

if __name__ == "__main__":
    main()