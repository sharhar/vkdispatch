Buffer Creation and Usage
=================

In vkdispatch, nearly all data is stored inside "buffers" (each wrapping an individual `VkBuffer <https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuffer.html>`_ object and all other objects needed to manage it). These are the equivalent of :class:`torch.Tensor` or :func:`wp.array` in :class:`warp-lang`.

However, unlike :class:`torch.Tensor` or :func:`wp.array`, vkdispatch buffers are by default multi-device. This means that when a vkdispatch buffer is allocated on a multi-device or multi-queue context, multiple :class:`VkBuffer`'s are allocated (one for each queue on each device). This architecture has the benefit of greatly simplfying multi-GPU programs, since all buffers can be assumed to exist on all devices and all queues. 

Allocating Buffers
---------------------

To allocate a buffer, you can use the constructor of the :class:`vkdispatch.Buffer` class. 


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
   2.  We use the :func:`vkdispatch.asbuffer` function to upload the numpy array to a vkdispatch buffer.
   3.  :func:`vkdispatch.Buffer.read` retrieves data back from the GPU to the CPU. The number provided as an argument to the function is the queue index to read from. For a simple context with one device and one queue, there is only 1 queue, so we read from index 0. If the index is ommited the function returns a python list of the contents of all buffers on all queues and devices.

Buffer Class API Reference
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

.. autofunction:: vkdispatch.asbuffer

