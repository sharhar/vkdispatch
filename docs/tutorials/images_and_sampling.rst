Images and Sampling
===================

Buffers are the default data container in vkdispatch, but image objects are available
for texture-like sampling workflows.

Image Types
-----------

vkdispatch provides:

* ``vd.Image1D``
* ``vd.Image2D``
* ``vd.Image2DArray``
* ``vd.Image3D``

Each image supports host-side ``write(...)`` and ``read(...)`` as well as shader-side
sampling through ``image.sample()``.

Basic Upload/Download Example
-----------------------------

.. code-block:: python

   import numpy as np
   import vkdispatch as vd

   data = np.sin(
       np.array([[i / 8 + j / 17 for i in range(64)] for j in range(64)])
   ).astype(np.float32)

   img = vd.Image2D(data.shape, vd.float32)
   img.write(data)

   roundtrip = img.read(0)
   print(np.allclose(roundtrip, data))

Sampling in a Shader
--------------------

Use codegen image argument types (``Img1``, ``Img2``, ``Img3``) inside ``@vd.shader``:

.. code-block:: python

   import vkdispatch.codegen as vc
   from vkdispatch.codegen.abreviations import *

   upscale = 4
   out = vd.Buffer((data.shape[0] * upscale, data.shape[1] * upscale), vd.float32)

   @vd.shader("out.size")
   def sample_2d(out: Buff[f32], src: Img2[f32], scale: Const[f32]):
       tid = vc.global_invocation_id().x
       ij = vc.ravel_index(tid, out.shape)
       uv = vc.new_vec2_register(ij.y, ij.x) / scale
       out[tid] = src.sample(uv).x

   sample_2d(out, img.sample(), float(upscale))
   sampled = out.read(0)

``img.sample()`` creates a sampler object with configurable filtering/address modes.

Sampler Configuration
---------------------

You can override sampling behavior:

.. code-block:: python

   sampler = img.sample(
       mag_filter=vd.Filter.LINEAR,
       min_filter=vd.Filter.LINEAR,
       address_mode=vd.AddressMode.CLAMP_TO_EDGE,
   )

   sample_2d(out, sampler, float(upscale))

Image API Reference
-------------------

See the :doc:`Full Python API Reference <../python_api>` for complete API details on:

* ``vkdispatch.Image``, ``vkdispatch.Image1D``, ``vkdispatch.Image2D``
* ``vkdispatch.Image2DArray``, ``vkdispatch.Image3D``
* ``vkdispatch.Sampler``, ``vkdispatch.Filter``
* ``vkdispatch.AddressMode``, ``vkdispatch.BorderColor``
