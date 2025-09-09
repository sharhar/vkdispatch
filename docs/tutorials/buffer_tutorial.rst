Buffer Creation and Usage
=================

The Buffer system is the heart of vkdispatch. All GPU memory operations
go through Buffer objects.

.. note::
   Always use BufferBuilder to create buffers - direct Buffer construction
   is not supported.

Buffer Class
------------

.. autoclass:: vkdispatch.Buffer
   :members: __init__, _destroy, write, read
   :show-inheritance:

   **Location:** vkdispatch.base.Buffer

   **Example Usage:**
   
   .. code-block:: python
   
      buffer = vd.Buffer((1000000,), vd.float32)
      buffer.write(my_data)
      result = buffer.read()


Your First GPU Buffer
---------------------



.. code-block:: python
   
   import vkdispatch as vd
   import numpy as np

   # Create a simple numpy array
   cpu_data = np.arange(16, dtype=np.int32)
   print(f"Original CPU data: {cpu_data}")

   # Create a GPU buffer
   gpu_buffer = vd.asbuffer(cpu_data)

   # Read data back from GPU to CPU to verify
   downloaded_data = gpu_buffer.read(0)
   print(f"Data downloaded from GPU: {downloaded_data.flatten()}")

   # Expected Output:
   # Original CPU data: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
   # Data downloaded from GPU: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]


.. admonition:: What's happening here?
   :class: tip

   1.  We import `vkdispatch` and `numpy` (a common dependency for numerical data).
   2.  A `BufferBuilder` is used to define the characteristics of our GPU buffer (size, usage).
   3.  `buffer.upload()` transfers data from your CPU's memory to the GPU.
   4.  `buffer.download()` retrieves data back from the GPU to the CPU.
   5.  Error checking is crucial in GPU programming, so `check_for_errors()` ensures operations completed successfully.


Buffer Builder
--------------

.. autoclass:: vkdispatch.BufferBuilder
   :members:
   :show-inheritance: