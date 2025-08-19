Getting Started with vkdispatch
===============================

Welcome to vkdispatch! This guide will help you install the library and run your first GPU-accelerated code.

.. note::
   vkdispatch requires a Vulkan-compatible GPU and drivers installed on your system.
   Please ensure your system meets these requirements before proceeding.

Installation
------------

You can install `vkdispatch` directly from PyPI using `pip`. We recommend
using a `virtual environment`_ for your projects.

.. code-block:: bash

   # Create a virtual environment (optional, but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

   # Install vkdispatch
   pip install vkdispatch

.. _virtual environment: https://docs.python.org/3/library/venv.html

Verifying Your Installation
---------------------------

To ensure `vkdispatch` is installed correctly and can detect your GPU,
run this simple Python script:

.. code-block:: python

   import vkdispatch

   # Initialize the vkdispatch context
   vkdispatch.initialize()

   # Get information about available GPU devices
   devices = vkdispatch.get_devices()

   if devices:
       print("vkdispatch successfully initialized!")
       print(f"Found {len(devices)} Vulkan-compatible GPUs:")
       for i, dev_info in enumerate(devices):
           print(f"  Device {i}: {dev_info.name} (Type: {dev_info.device_type})")
   else:
       print("No Vulkan-compatible GPUs found or vkdispatch failed to initialize.")

   # Clean up the context
   vkdispatch.get_context().deinitialize()

If the installation was successful, you should see output listing your GPU(s).

Your First GPU Buffer
---------------------

Let's create a simple GPU buffer and fill it with data.

.. literalinclude:: ../examples/first_buffer.py
   :language: python
   :linenos:
   :caption: examples/first_buffer.py

.. raw:: html

.. code-block:: text

      # Expected Output:
      # Data uploaded to GPU: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
      # Data downloaded from GPU: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
   

.. admonition:: What's happening here?
   :class: tip

   1.  We import `vkdispatch` and `numpy` (a common dependency for numerical data).
   2.  A `BufferBuilder` is used to define the characteristics of our GPU buffer (size, usage).
   3.  `buffer.upload()` transfers data from your CPU's memory to the GPU.
   4.  `buffer.download()` retrieves data back from the GPU to the CPU.
   5.  Error checking is crucial in GPU programming, so `check_for_errors()` ensures operations completed successfully.

Next Steps
----------

Now that you've got `vkdispatch` up and running, consider exploring:

*   **Public API Reference:** Our curated guide to the most commonly used classes and functions.
*   **Full Python API Reference:** A comprehensive list of all Python-facing components.
*   **C++/Cython API Reference:** Dive deep into the backend details.

Happy GPU programming!

.. seealso::

   :doc:`public_api/index`
      Start here for a guided tour of core features.
   :doc:`buffer_management`
      Detailed information on working with GPU buffers.