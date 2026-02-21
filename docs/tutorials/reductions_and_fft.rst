Reductions and FFT Workflows
============================

This page covers common high-level numeric workflows in vkdispatch:

* reductions with ``vd.reduce``
* Fourier transforms with ``vd.fft``
* VkFFT-backed transforms with ``vd.vkfft``

Reduction Basics
----------------

Use ``@vd.reduce.reduce`` for pure binary reductions:

.. code-block:: python

   import numpy as np
   import vkdispatch as vd
   from vkdispatch.codegen.abreviations import *

   @vd.reduce.reduce(0)
   def sum_reduce(a: f32, b: f32) -> f32:
       return a + b

   arr = np.random.rand(4096).astype(np.float32)
   buf = vd.asbuffer(arr)
   out = sum_reduce(buf).read(0)

   print("GPU sum:", float(out[0]))
   print("CPU sum:", float(arr.sum(dtype=np.float32)))

Mapped Reductions
-----------------

Use ``@vd.reduce.map_reduce`` when you want a map stage before reduction:

.. code-block:: python

   import vkdispatch.codegen as vc

   @vd.reduce.map_reduce(vd.reduce.SubgroupAdd)
   def l2_energy_map(buffer: Buff[f32]) -> f32:
       idx = vd.reduce.mapped_io_index()
       v = buffer[idx]
       return v * v

   energy_buf = l2_energy_map(buf)
   energy = energy_buf.read(0)[0]

This pattern is useful for sums of transformed values (norms, weighted sums, etc.).

FFT with ``vd.fft``
-------------------

The ``vd.fft`` module dispatches vkdispatch-generated FFT shaders.

.. code-block:: python

   complex_signal = (
       np.random.rand(256) + 1j * np.random.rand(256)
   ).astype(np.complex64)

   fft_buf = vd.asbuffer(complex_signal)

   vd.fft.fft(fft_buf)
   freq = fft_buf.read(0)

   vd.fft.ifft(fft_buf)
   recovered = fft_buf.read(0)

   print(np.allclose(recovered, complex_signal, atol=1e-3))

Real FFT (RFFT) helpers:

.. code-block:: python

   real_signal = np.random.rand(512).astype(np.float32)
   rbuf = vd.asrfftbuffer(real_signal)

   vd.fft.rfft(rbuf)
   spectrum = rbuf.read_fourier(0)

   vd.fft.irfft(rbuf)
   restored = rbuf.read_real(0)

   print(np.allclose(restored, real_signal, atol=1e-3))

FFT with ``vd.vkfft``
---------------------

``vd.vkfft`` exposes a similar API but routes operations through VkFFT plan objects
with internal plan caching.

.. code-block:: python

   vkfft_buf = vd.asbuffer(complex_signal.copy())
   vd.vkfft.fft(vkfft_buf)
   vd.vkfft.ifft(vkfft_buf)
   print(np.allclose(vkfft_buf.read(0), complex_signal, atol=1e-3))

After large parameter sweeps, clearing cached plans can be helpful:

.. code-block:: python

   vd.vkfft.clear_plan_cache()
   vd.fft.cache_clear()

Convolution Helpers
-------------------

vkdispatch also includes FFT-based convolution helpers:

* ``vd.fft.convolve`` / ``vd.fft.convolve2D`` / ``vd.fft.convolve2DR``
* ``vd.vkfft.convolve2D`` and ``vd.vkfft.transpose_kernel2D``

These APIs are most useful when you repeatedly convolve signals/images with known
kernel layouts.

Reduction and FFT API Reference
-------------------------------

See the :doc:`Full Python API Reference <../python_api>` for complete API details on:

* ``vkdispatch.reduce``
* ``vkdispatch.fft``
* ``vkdispatch.vkfft``
