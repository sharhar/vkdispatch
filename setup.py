import os
import platform
import re
import subprocess
from pathlib import Path

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext

try:
    from packaging.version import Version
except ImportError:
    print("Warning: 'packaging' not found; version comparisons might be less accurate.")
    from distutils.version import LooseVersion as Version


BUILD_TARGET_FULL = "full"
BUILD_TARGET_CORE = "core"
BUILD_TARGET_NATIVE = "native"
BUILD_TARGET_META = "meta"
VALID_BUILD_TARGETS = {
    BUILD_TARGET_FULL,
    BUILD_TARGET_CORE,
    BUILD_TARGET_NATIVE,
    BUILD_TARGET_META,
}


def get_build_target() -> str:
    target = os.environ.get("VKDISPATCH_BUILD_TARGET", BUILD_TARGET_FULL).strip().lower()
    if target not in VALID_BUILD_TARGETS:
        valid = ", ".join(sorted(VALID_BUILD_TARGETS))
        raise RuntimeError(
            f"Invalid VKDISPATCH_BUILD_TARGET={target!r}. Expected one of: {valid}"
        )
    return target


BUILD_TARGET = get_build_target()

proj_root = Path(__file__).resolve().parent
system = platform.system()
molten_vk_path = "./deps/MoltenVK/MoltenVK/MoltenVK/static/MoltenVK.xcframework/macos-arm64_x86_64/"
vulkan_sdk_root = os.environ.get("VULKAN_SDK")


def read_version() -> str:
    init_path = proj_root / "vkdispatch" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {init_path}")
    return match.group(1)


def read_readme() -> str:
    return (proj_root / "README.md").read_text(encoding="utf-8")


VERSION = read_version()

COMMON_CLASSIFIERS = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Development Status :: 3 - Alpha",
]

COMMON_PROJECT_URLS = {
    "Homepage": "https://github.com/sharhar/vkdispatch",
    "Issues": "https://github.com/sharhar/vkdispatch/issues",
}

COMMON_EXTRAS = {
    "cuda": ["cuda-python", "numpy"],
    "opencl": ["pyopencl", "numpy"],
    "numpy": ["numpy"],
}


def parse_compiler_version(version_output):
    if not isinstance(version_output, str):
        return None

    clang_match = re.search(r"clang version ([^\s]+)", version_output)
    gcc_match = re.search(
        r"gcc.+?([\d.]+(?:-[a-zA-Z0-9]+)?)", version_output, re.IGNORECASE
    )

    match = clang_match or gcc_match
    if not match:
        return None

    try:
        return Version(match.group(1))
    except Exception as exc:
        print(f"Invalid version: {exc}")
        return None


def detect_unix_compiler(compiler_exe):
    try:
        version_output = subprocess.check_output(
            [compiler_exe, "--version"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        if "clang" in version_output:
            return "clang", parse_compiler_version(version_output)
        if "gcc" in version_output or "Free Software Foundation" in version_output:
            return "gcc", parse_compiler_version(version_output)
        return "unknown", None
    except Exception:
        return "unknown", None


class CustomBuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        print(f"Detected compiler type: {compiler_type}")

        if compiler_type == "unix":
            print(f"Detected compiler: {self.compiler.compiler}")
            compiler_family, version = detect_unix_compiler(self.compiler.compiler[0])
            print(f"Detected compiler family: {compiler_family}")
            print(f"Detected compiler version: {version}")

            if version is not None:
                for ext in self.extensions:
                    if compiler_family == "clang" and version < Version("9.0"):
                        ext.libraries.append("c++fs")
                    elif compiler_family == "gcc" and version < Version("9.1"):
                        ext.libraries.append("stdc++fs")
                    else:
                        print(
                            "WARNING: Unknown compiler family, not adding filesystem library"
                        )

        super().build_extensions()


def append_to_sources(prefix, source_list, out_sources):
    for source in source_list:
        out_sources.append(prefix + source)


def build_native_extension():
    platform_library_dirs = []
    platform_define_macros = []
    platform_link_libraries = []
    platform_extra_link_args = []
    platform_extra_compile_args = (
        ["/W3", "/GL", "/DNDEBUG", "/MD", "/EHsc", "/std:c++17"]
        if system == "Windows"
        else ["-O2", "-g", "-std=c++17"]
    )

    include_directories = [
        str(proj_root / "deps" / "VMA" / "include"),
        str(proj_root / "deps" / "volk"),
        str(proj_root / "deps" / "VkFFT" / "vkFFT"),
    ]

    if os.name == "posix":
        platform_extra_link_args.extend(["-g", "-O0", "-fno-omit-frame-pointer"])
        platform_link_libraries.extend(["dl", "pthread"])

    if vulkan_sdk_root is None:
        include_directories.extend(
            [
                str(proj_root / "include_ext"),
                str(proj_root / "deps" / "Vulkan-Headers" / "include"),
                str(proj_root / "deps" / "Vulkan-Utility-Libraries" / "include"),
                str(proj_root / "deps" / "glslang"),
                str(proj_root / "deps" / "glslang" / "glslang" / "Include"),
            ]
        )

        if system == "Darwin":
            platform_library_dirs.append(molten_vk_path)
            platform_link_libraries.append("MoltenVK")
            platform_extra_link_args.extend(
                [
                    "-framework",
                    "Metal",
                    "-framework",
                    "AVFoundation",
                    "-framework",
                    "AppKit",
                ]
            )
            platform_extra_compile_args.append("-mmacosx-version-min=10.15")
        else:
            platform_define_macros.append(("VKDISPATCH_USE_VOLK", 1))
    else:
        include_directories.extend(
            [
                vulkan_sdk_root + "/include",
                vulkan_sdk_root + "/include/utility",
                vulkan_sdk_root + "/include/glslang/Include",
            ]
        )

        platform_define_macros.append(("VKDISPATCH_USE_VOLK", 1))
        platform_define_macros.append(
            ("VKDISPATCH_LOADER_PATH", '"' + os.path.abspath(vulkan_sdk_root) + '/"')
        )

        platform_library_dirs.append(vulkan_sdk_root + "/lib")
        platform_link_libraries.extend(
            [
                "glslang",
                "SPIRV",
                "MachineIndependent",
                "GenericCodeGen",
                "SPIRV-Tools-opt",
                "SPIRV-Tools-link",
                "SPIRV-Tools-reduce",
                "SPIRV-Tools",
                "glslang-default-resource-limits",
            ]
        )

    sources = []
    sources.append("vkdispatch_native/wrapper.pyx")

    append_to_sources(
        "vkdispatch_native/",
        [
            "context/init.cpp",
            "context/context.cpp",
            "context/errors.cpp",
            "context/handles.cpp",
            "objects/buffer.cpp",
            "objects/image.cpp",
            "objects/command_list.cpp",
            "objects/descriptor_set.cpp",
            "stages/stage_fft.cpp",
            "stages/stage_compute.cpp",
            "queue/queue.cpp",
            "queue/signal.cpp",
            "queue/work_queue.cpp",
            "queue/barrier_manager.cpp",
            "libs/VMAImpl.cpp",
            "libs/VolkImpl.cpp",
        ],
        sources,
    )

    if vulkan_sdk_root is None:
        append_to_sources(
            "deps/glslang/glslang/",
            [
                "CInterface/glslang_c_interface.cpp",
                "GenericCodeGen/CodeGen.cpp",
                "GenericCodeGen/Link.cpp",
                "MachineIndependent/glslang_tab.cpp",
                "MachineIndependent/attribute.cpp",
                "MachineIndependent/Constant.cpp",
                "MachineIndependent/iomapper.cpp",
                "MachineIndependent/InfoSink.cpp",
                "MachineIndependent/Initialize.cpp",
                "MachineIndependent/IntermTraverse.cpp",
                "MachineIndependent/Intermediate.cpp",
                "MachineIndependent/ParseContextBase.cpp",
                "MachineIndependent/ParseHelper.cpp",
                "MachineIndependent/PoolAlloc.cpp",
                "MachineIndependent/RemoveTree.cpp",
                "MachineIndependent/Scan.cpp",
                "MachineIndependent/ShaderLang.cpp",
                "MachineIndependent/SpirvIntrinsics.cpp",
                "MachineIndependent/SymbolTable.cpp",
                "MachineIndependent/Versions.cpp",
                "MachineIndependent/intermOut.cpp",
                "MachineIndependent/limits.cpp",
                "MachineIndependent/linkValidate.cpp",
                "MachineIndependent/parseConst.cpp",
                "MachineIndependent/reflection.cpp",
                "MachineIndependent/preprocessor/Pp.cpp",
                "MachineIndependent/preprocessor/PpAtom.cpp",
                "MachineIndependent/preprocessor/PpContext.cpp",
                "MachineIndependent/preprocessor/PpScanner.cpp",
                "MachineIndependent/preprocessor/PpTokens.cpp",
                "MachineIndependent/propagateNoContraction.cpp",
                "ResourceLimits/ResourceLimits.cpp",
                "ResourceLimits/resource_limits_c.cpp",
            ],
            sources,
        )

        append_to_sources(
            "deps/glslang/SPIRV/",
            [
                "GlslangToSpv.cpp",
                "InReadableOrder.cpp",
                "Logger.cpp",
                "SpvBuilder.cpp",
                "SpvPostProcess.cpp",
                "doc.cpp",
                "SpvTools.cpp",
                "disassemble.cpp",
                "CInterface/spirv_c_interface.cpp",
            ],
            sources,
        )

    return Extension(
        "vkdispatch_vulkan_native",
        sources=sources,
        language="c++",
        define_macros=platform_define_macros,
        library_dirs=platform_library_dirs,
        libraries=platform_link_libraries,
        extra_compile_args=platform_extra_compile_args,
        extra_link_args=platform_extra_link_args,
        include_dirs=include_directories,
    )


def base_setup_kwargs():
    return {
        "version": VERSION,
        "author": "Shahar Sandhaus",
        "author_email": "shahar.sandhaus@gmail.com",
        "description": "Python metaprogramming for GPU compute, with runtime-generated kernels, FFTs, and reductions.",
        "long_description": read_readme(),
        "long_description_content_type": "text/markdown",
        "python_requires": ">=3.6",
        "classifiers": COMMON_CLASSIFIERS,
        "project_urls": COMMON_PROJECT_URLS,
        "zip_safe": False,
    }


def core_packages():
    return find_packages(include=["vkdispatch", "vkdispatch.*"])


def setup_for_target(target: str):
    kwargs = base_setup_kwargs()

    if target == BUILD_TARGET_FULL:
        kwargs.update(
            {
                "name": "vkdispatch",
                "packages": core_packages(),
                "install_requires": ["setuptools>=59.0"],
                "extras_require": {
                    "cli": ["Click"],
                    **COMMON_EXTRAS,
                },
                "entry_points": {
                    "console_scripts": [
                        "vdlist=vkdispatch.cli:cli_entrypoint",
                    ]
                },
                "ext_modules": [build_native_extension()],
                "cmdclass": {"build_ext": CustomBuildExt},
            }
        )
        return kwargs

    if target == BUILD_TARGET_CORE:
        kwargs.update(
            {
                "name": "vkdispatch-core",
                "packages": core_packages(),
                "install_requires": ["setuptools>=59.0"],
                "extras_require": {
                    "cli": ["Click"],
                    **COMMON_EXTRAS,
                },
                "entry_points": {
                    "console_scripts": [
                        "vdlist=vkdispatch.cli:cli_entrypoint",
                    ]
                },
            }
        )
        return kwargs

    if target == BUILD_TARGET_NATIVE:
        kwargs.update(
            {
                "name": "vkdispatch-vulkan-native",
                "packages": [],
                "py_modules": [],
                "install_requires": [],
                "ext_modules": [build_native_extension()],
                "cmdclass": {"build_ext": CustomBuildExt},
            }
        )
        return kwargs

    if target == BUILD_TARGET_META:
        kwargs.update(
            {
                "name": "vkdispatch",
                "packages": [],
                "py_modules": [],
                "install_requires": [
                    f"vkdispatch-core=={VERSION}",
                    f"vkdispatch-vulkan-native=={VERSION}",
                ],
                "extras_require": {
                    "cli": ["Click"],
                    **COMMON_EXTRAS,
                },
            }
        )
        return kwargs

    raise AssertionError(f"Unhandled build target: {target}")


setup(**setup_for_target(BUILD_TARGET))
