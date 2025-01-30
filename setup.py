import os
import platform

from setuptools import Extension
from setuptools import setup

system = platform.system()

proj_root = os.path.abspath(os.path.dirname(__file__))
molten_vk_path = "./deps/MoltenVK/MoltenVK/MoltenVK/static/MoltenVK.xcframework/macos-arm64_x86_64/"
vulkan_sdk_root = os.environ.get('VULKAN_SDK')

platform_name_dict = {
    "Darwin": "MACOS",
    "Windows": "WINDOWS",
    "Linux": "LINUX"
}

platform_library_dirs = []
platform_define_macros = [(f"__VKDISPATCH_PLATFORM_{platform_name_dict[system]}__", 1), ("LOG_VERBOSE_ENABLED", 1)]
platform_link_libraries = []
platform_extra_link_args = []
platform_extra_compile_args = (
    ["/W3", "/GL", "/DNDEBUG", "/MD", "/EHsc", "/std:c++17"]
    if system == "Windows"
    else ["-g", "-std=c++17"]
)

include_directories = [
    proj_root + "/deps/VMA/include",
    proj_root + "/deps/volk",
    proj_root + "/deps/VkFFT/vkFFT",
]

if os.name == "posix":
    platform_extra_link_args.append("-g")
    platform_link_libraries.extend(["dl", "pthread"])

if vulkan_sdk_root is None:
    include_directories.extend([
        proj_root + "/include_ext",
        proj_root + "/deps/Vulkan-Headers/include",
        proj_root + "/deps/Vulkan-Utility-Libraries/include",
        proj_root + "/deps/glslang",
        proj_root + "/deps/glslang/glslang/Include",
    ])

    if system == "Darwin":
        platform_library_dirs.append(molten_vk_path)
        platform_link_libraries.append("MoltenVK")
        platform_extra_link_args.extend([
            "-framework", "Metal",
            "-framework", "AVFoundation",
            "-framework", "AppKit"
        ])
        platform_extra_compile_args.append("-mmacosx-version-min=10.15")
    else:
        platform_define_macros.append(("VKDISPATCH_USE_VOLK", 1))
else:
    include_directories.extend([
        vulkan_sdk_root + '/include',
        vulkan_sdk_root + '/include/utility',
        vulkan_sdk_root + '/include/glslang/Include',
    ])

    platform_define_macros.append(("VKDISPATCH_USE_VOLK", 1))
    platform_define_macros.append(("VKDISPATCH_LOADER_PATH", '"' + os.path.abspath(f"{vulkan_sdk_root}") + '/"'))

    #if os.name == "posix":
    #    platform_link_libraries.append("vulkan")
    #else:
    #    platform_link_libraries.append("vulkan-1")

    platform_library_dirs.append(vulkan_sdk_root + '/lib')

    platform_link_libraries.extend([
        "glslang",
        "SPIRV", 
        "MachineIndependent",
        "GenericCodeGen",
        "SPIRV-Tools-opt",
        "SPIRV-Tools-link", 
        "SPIRV-Tools-reduce",
        "SPIRV-Tools",
        "glslang-default-resource-limits"
    ])


sources = []

def append_to_sources(prefix, source_list):
    global sources

    for source in source_list:
        sources.append(prefix + source)


sources.append("vkdispatch_native/wrappers/wrapper.pyx")

append_to_sources("vkdispatch_native/src/", [
    "init.cpp",
    "context.cpp",
    "conditional.cpp",
    "buffer.cpp",
    "image.cpp",
    "command_list.cpp",
    "stage_transfer.cpp",
    "stage_fft.cpp",
    "stage_compute.cpp",
    "descriptor_set.cpp",
    "stream.cpp",
    "errors.cpp",
    "signal.cpp",
    "work_queue.cpp",
    "VMAImpl.cpp",
    "VolkImpl.cpp"
])

if vulkan_sdk_root is None:
    append_to_sources("deps/glslang/glslang/", [
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
        "ResourceLimits/resource_limits_c.cpp"
    ])

    append_to_sources("deps/glslang/SPIRV/", [
        "GlslangToSpv.cpp",
        "InReadableOrder.cpp",
        "Logger.cpp",
        "SpvBuilder.cpp",
        "SpvPostProcess.cpp",
        "doc.cpp",
        "SpvTools.cpp",
        "disassemble.cpp",
        "CInterface/spirv_c_interface.cpp"
    ])

setup(
    name="vkdispatch",
    packages=["vkdispatch", "vkdispatch.base", "vkdispatch.codegen", "vkdispatch.execution_pipeline", "vkdispatch.shader_generation"],
    ext_modules=[
        Extension(
            "vkdispatch_native",
            sources=sources,
            language="c++",
            define_macros=platform_define_macros,
            library_dirs=platform_library_dirs,
            libraries=platform_link_libraries,
            extra_compile_args=platform_extra_compile_args,
            extra_link_args=platform_extra_link_args,
            include_dirs=include_directories,
        )
    ],
    version="0.0.18",
    zip_safe=False,
)
