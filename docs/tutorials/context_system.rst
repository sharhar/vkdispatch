Initialization and Context Creation
==============

In vkdispatch, all operations are handled by a global context object which keeps track of things behind the scenes. By default, any function call into the vkdispatch API (aside from the :func:`vd.shader <vkdispatch.shader>` decorator) invokes the creation of a context with default settings. This means that **this page can be skipped if you have no need to customize context parameters.**

However, you can control the context creation process for one of many reasons:

* To utlize more than one device or queue, for full control over GPU resources
* To enable debugging features such as printing to stdout from shaders
* To select the logging level to use by default


Initialization
--------------

The first part of any vkdispatch program is initialization. This step is distinct from context creation since it controls the creation of the `VkInstance <https://registry.khronos.org/vulkan/specs/latest/man/html/VkInstance.html>`_ object, which is required to be able to list the number and type of devices in the system. Since this information may be useful for context creation, the step of initialization and context creation is seperate.

To create a context you must call :func:`vd.initialize() <vkdispatch.initialize>` before any other call
to a vkdispatch API, a few examples are provided below:

.. code-block:: python

   import vkdispatch as vd

   # Enables debug mode, which allows for printing from shaders
   vd.initialize(debug_mode=True)

   # Sets the environment variable `VK_LOADER_DEBUG` to 'all'. 
   # This enables debug log outputs for the vulkan loader, which
   # can be useful for debugging driver loading issues 
   vd.initialize(loader_debug_logs=True)

   # Sets the default logging level to INFO, which enables detailed printouts
   # of internal vkdispatch operations, useful for debugging internal issues.
   vd.initialize(log_level=vd.LogLevel.INFO)

.. note::
   The debug_mode flag enables the `VK_EXT_debug_utils <https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_debug_utils.html>`_ vulkan extension and singals the creation of a `VkDebugUtilsMessengerEXT <https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessengerEXT.html>`_ object. This allows for printing from shaders, but also significantly reduces performance by introducing runtime debugging tools. Therefore, it is recommended this option remain off unless needed for in shader debugging.

Initialization API Reference
------------------

.. autofunction:: vkdispatch.initialize

.. .. autoclass:: vkdispatch.LogLevel
..    :members: VERBOSE, INFO, WARNING, ERROR
..    :show-inheritance:

Context Management
------------------

After the initialization stage, vkdispatch must create a context to be used. By default, this context will be created when the vkdispatch API is first used (like the initialization). The default context is configured to only use one GPU with one queue, to enable multi-device contexts, the user must manually call :func:`vd.make_context() <vkdispatch.make_context>` before calling any other vkdispatch API:

.. code-block:: python

   import vkdispatch as vd

   # Call the initilization first if you want to modify the instance settings
   # vd.initialize(...)

   # To use all available devices, we create the context with the `multi_device` flag
   vd.make_context(multi_device=True)

   # To allocate multiple queues per device for maximum utilization we use the `multi_queue` flag
   vd.make_context(multi_queue=True)

   # To utilize all devices with multiple queues, we use both flags
   vd.make_context(multi_device=True, multi_queue=True)

Context API Reference
------------------
.. autofunction:: vkdispatch.make_context

