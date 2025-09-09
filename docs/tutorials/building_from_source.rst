Building from Source
====================

This page is for contributors and power users who want to **clone the repository and
modify vkdispatch**, or for platforms where a prebuilt wheel is not available and pip falls
back to a source build.

Who should use this?
--------------------
- You plan to edit vkdispatch and need an **editable/development install**.
- You’re on a **non-standard architecture** (e.g., non-Apple ARM/aarch64 or niche OS),
  where pip cannot find a prebuilt wheel.
- You want to **rebuild a wheel locally** for testing or distribution.

Prerequisites
-------------
Most builds succeed with just a modern compiler and Python. For clarity:

- **Compilation Requirements**:
  - A **C++17-capable compiler**
    - Linux: GCC ≥ 9 or Clang ≥ 10
    - macOS: Xcode Command Line Tools (``xcode-select --install``)
    - Windows: Microsoft C++ Build Tools or Visual Studio 2019+ (x64)
  - **Python development headers** 

    .. code-block:: bash

        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y build-essential python3-dev

        # Fedora/RHEL
        sudo dnf groupinstall -y "Development Tools"
        sudo dnf install -y python3-devel

Quick start (clone → editable install)
--------------------------------------
Use an editable install to iterate on the code without reinstalling each change.

.. code-block:: bash

   # 1) Clone your fork or the upstream repo (replace with your URL)
   git clone https://github.com/sharhar/vkdispatch.git
   cd vkdispatch

   # Download source code of dependencies
   python fetch_dependencies.py

   # 3) Create/activate a clean environment (recommended)
   python -m venv .venv && . .venv/bin/activate   # on macOS/Linux
   # .venv\Scripts\activate                       # on Windows (PowerShell/CMD)

   # 4) Install in editable mode
   pip install -e .

Build a wheel locally (optional)
--------------------------------
If you prefer a built artifact (e.g., CI, packaging, testing import behavior):

.. code-block:: bash

   # Build a wheel into ./dist
   pip wheel . -w dist

   # Or using the 'build' frontend (creates sdist + wheel under ./dist)
   python -m build

Troubleshooting
---------------
- **error: Python.h: No such file or directory**  
  Install your distro’s Python headers (``python3-dev`` / ``python3-devel``).

- **error: Missing header**
  Fetch the source dependencies by calling ``python3 fetch_dependencies.py``.

Clean rebuild tips
------------------
.. code-block:: bash

   # Remove previous builds/artifacts and reinstall verbosely
   pip uninstall -y vkdispatch
   rm -rf build/ dist/ *.egg-info
   pip install -e . -v
