Data Types Overview
=====================

In vkdispatch, there are a number of different datatypes that can be used to store data in buffers and images and to process data in shaders. These data types come in 3 formats (for now, all 32 bit):

 * signed int (e.g. :class:`int32 <vkdispatch.int32>` and :class:`ivec2 <vkdispatch.ivec2>`)
 * unsigned int (e.g. :class:`uint32 <vkdispatch.uint32>` and :class:`uvec3 <vkdispatch.uvec3>`)
 * floating point (e.g. :class:`float32 <vkdispatch.float32>` and :class:`vec4 <vkdispatch.vec4>`)

In the near future 64 bit and 16 bit types will be added.

They also come in the following shapes:

 * Scalars
 * Complex Number (internally represented as `vec2` in GLSL shaders)
 * Vectors (of size 2, 3, and 4)
 * Matricies (only :class:`vkdispatch.float32` at 2x2 and 4x4)

Data Type API Reference
---------------------

.. autofunction:: vkdispatch.is_dtype

.. autofunction:: vkdispatch.is_scalar

.. autofunction:: is_complex

.. autofunction:: vkdispatch.is_vector

.. autofunction:: vkdispatch.is_matrix

.. autofunction:: vkdispatch.from_numpy_dtype

.. autofunction:: vkdispatch.to_numpy_dtype

.. autoclass:: vkdispatch.dtype

.. autoclass:: vkdispatch.int32

.. autoclass:: vkdispatch.uint32

.. autoclass:: vkdispatch.float32

.. autoclass:: vkdispatch.complex64

.. autoclass:: vkdispatch.vec2

.. autoclass:: vkdispatch.vec3

.. autoclass:: vkdispatch.vec4

.. autoclass:: vkdispatch.ivec2

.. autoclass:: vkdispatch.ivec3

.. autoclass:: vkdispatch.ivec4

.. autoclass:: vkdispatch.uvec2

.. autoclass:: vkdispatch.uvec3

.. autoclass:: vkdispatch.uvec4

.. autoclass:: vkdispatch.mat2

.. autoclass:: vkdispatch.mat4