Context System
==============

.. This page is now its own HTML file

In vkdispatch, all operations are handled by a global context object which keeps
track of things behind the scenes. By default, any function call into the vkdispatch
API (aside from the vd.shader decorator) invokes the creation of a context with default
settings. However, you can also control the context creation process for on of many reasons:

* To utlize more than one device or queue, for full control over GPU resources
* To enable debugging features such as printing to stdout from shaders
* To select the logging level to use by default


Initialization
--------------

The first part of any vkdispatch program is initialization. This step is distinct from
context creation since it controls the creation of the :class:`VkInstance` object, which
is required to be able to list the number and type of devices in the system. Since this
information may be useful for context creation, the step of initialization and context 
creation is seperate. To create a context you must call :class:`vkdispatch.initialize`
before any other call to a vkdispatch API. 


Context Management
------------------

Context API
------------------

.. autofunction:: vkdispatch.initialize

.. autofunction:: vkdispatch.get_devices

.. autofunction:: vkdispatch.make_context

.. autofunction:: vkdispatch.get_context