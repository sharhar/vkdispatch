# Getting Started / Installation

Welcome to **vkdispatch**! This guide will help you install the library and run your first GPU-accelerated code.

> **Note:** vkdispatch requires a Vulkan-compatible GPU and drivers installed on your system. Please ensure your system meets these requirements before proceeding.

## PyPI

The default installation method for `vkdispatch` is through PyPI (pip):

```bash
# Install the package
pip install vkdispatch
```

On mainstream platforms — Windows (x86_64), macOS (x86_64 and Apple Silicon/arm64), and Linux (x86_64) — pip will download a **prebuilt wheel** (built with `cibuildwheel` on GitHub Actions and tagged as *manylinux* where applicable), so no compiler is needed.

On less common platforms (e.g., non-Apple ARM or other niche architectures), pip may fall back to a **source build**, which takes a few minutes. See **[Building From Source](https://sharhar.github.io/vkdispatch/tutorials/building_from_source.html)** for toolchain requirements and developer-oriented instructions.  
*(Replace the link above with your actual GitHub Pages URL once deployed.)*

> **Tip:** If you see output like `Building wheel for vkdispatch (pyproject.toml)`, you’re compiling from source.

## Verifying Your Installation

To ensure `vkdispatch` is installed correctly and can detect your GPU, run:

```bash
# Quick device listing
vdlist

# If the above command is unavailable, try:
python3 -m vkdispatch
```

If the installation was successful, you should see output listing your GPU(s), for example:

```text
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
```

## Next Steps

- **[Tutorials](https://sharhar.github.io/vkdispatch/tutorials/index.html)** — our curated guide to common workflows and examples
- **[Full Python API Reference](https://sharhar.github.io/vkdispatch/python_api.html)** — comprehensive reference for Python-facing components

Happy GPU programming!
