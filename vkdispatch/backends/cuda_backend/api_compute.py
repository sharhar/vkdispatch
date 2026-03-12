from __future__ import annotations

from . import state as state
from .cuda_primitives import SourceModule, cuda
from .helpers import (
    activate_context,
    context_from_handle,
    new_handle,
    parse_kernel_params,
    parse_local_size,
    set_error,
    to_bytes,
)
from .state import CUDAComputePlan


def _nvrtc_compile_options(ctx):
    options = ["-w"]

    try:
        dev = cuda.Device(ctx.device_index)
        cc_major, cc_minor = dev.compute_capability()
        options.append(f"--gpu-architecture=sm_{int(cc_major)}{int(cc_minor)}")
    except Exception:
        pass

    return options


def stage_compute_plan_create(context, shader_source, bindings, pc_size, shader_name):
    ctx = context_from_handle(int(context))
    if ctx is None:
        return 0

    source_bytes = to_bytes(shader_source)
    shader_name_bytes = to_bytes(shader_name)
    source_text = source_bytes.decode("utf-8", errors="replace")

    try:
        with activate_context(ctx):
            module = SourceModule(
                source_text,
                no_extern_c=True,
                options=_nvrtc_compile_options(ctx),
            )
            function = module.get_function("vkdispatch_main")
    except Exception as exc:
        set_error(f"Failed to compile CUDA kernel '{shader_name_bytes.decode(errors='ignore')}': {exc}")
        return 0

    try:
        params = parse_kernel_params(source_text)
        local_size = parse_local_size(source_text)
    except Exception as exc:
        set_error(f"Failed to parse CUDA kernel metadata: {exc}")
        return 0

    plan = CUDAComputePlan(
        context_handle=int(context),
        shader_source=source_bytes,
        bindings=[int(x) for x in bindings],
        shader_name=shader_name_bytes,
        module=module,
        function=function,
        local_size=local_size,
        params=params,
        pc_size=int(pc_size),
    )

    return new_handle(state.compute_plans, plan)


def stage_compute_plan_destroy(plan):
    if plan is None:
        return
    state.compute_plans.pop(int(plan), None)
