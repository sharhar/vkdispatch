Getting Started
===============================

Welcome to vkdispatch! This guide will help you install the library and run your first GPU-accelerated code.

.. note::
   vkdispatch requires a Vulkan-compatible GPU and drivers installed on your system.
   Please ensure your system meets these requirements before proceeding.

Installation
---------------------------

The default installation method for `vkdispatch` is through PyPI (pip):

.. code-block:: bash

   # Install the package
   pip install vkdispatch

On mainstream platforms — Windows (x86_64), macOS (x86_64 and Apple Silicon/arm64),
and Linux (x86_64) — pip will download a **prebuilt wheel** (built with `cibuildwheel`
on GitHub Actions and tagged as *manylinux* where applicable), so no compiler is needed.

On less common platforms (e.g., non-Apple ARM or other niche architectures), pip may
fall back to a **source build**, which takes a few minutes. See :doc:`Building From Source<tutorials/building_from_source>`
for toolchain requirements and developer-oriented instructions.

.. note::
   If you see output like ``Building wheel for vkdispatch (pyproject.toml)``,
   you’re compiling from source.

Verifying Your Installation
---------------------------

To ensure `vkdispatch` is installed correctly and can detect your GPU,
run this simple Python script:

.. code-block:: bash
   
   # Run the example script to verify installation
   vdlist

   # If the above command fails, you can try this alternative
   python3 -m vkdispatch

If the installation was successful, you should see output listing your GPU(s) which may look something like this:

.. code-block:: text

   Device 0: Apple M2 Pro
        Vulkan Version: 1.2.283
        Device Type: Integrated GPU

        Features:
                Float32 Atomic Add: True

        Properties:
                64-bit Float Support: False
                16-bit Float Support: True
                64-bit Int Support: True
                16-bit Int Support: True
                Max Push Constant Size: 4096 bytes
                Subgroup Size: 32
                Max Compute Shared Memory Size: 32768

        Queues:
                0 (count=1, flags=0x7): Graphics | Compute
                1 (count=1, flags=0x7): Graphics | Compute
                2 (count=1, flags=0x7): Graphics | Compute
                3 (count=1, flags=0x7): Graphics | Compute



Next Steps
----------

Now that you've got `vkdispatch` up and running, consider exploring the following:

*   :doc:`Tutorials<tutorials/index>`: Our curated guide to the most commonly used classes and functions.
*   :doc:`Full Python API Reference<python_api>`: A comprehensive list of all Python-facing components.

Happy GPU programming!