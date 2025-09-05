Buffer Tutorial
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



Buffer Builder
--------------

.. autoclass:: vkdispatch.BufferBuilder
   :members:
   :show-inheritance: