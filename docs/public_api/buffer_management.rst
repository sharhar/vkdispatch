Buffer Management
=================

The Buffer system is the heart of vkdispatch. All GPU memory operations
go through Buffer objects.

.. note::
   Always use BufferBuilder to create buffers - direct Buffer construction
   is not supported.

Buffer Class
------------

.. autoclass:: vkdispatch.Buffer
   :members: upload, download, size, resize, clear
   :show-inheritance:

   **Example Usage:**
   
   .. code-block:: python
   
      buffer = BufferBuilder().size(1024).build()
      buffer.upload(my_data)
      result = buffer.download()

Buffer Builder
--------------

.. autoclass:: vkdispatch.BufferBuilder
   :members:
   :show-inheritance: