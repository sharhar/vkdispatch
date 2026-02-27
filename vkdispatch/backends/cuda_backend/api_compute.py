from __future__ import annotations

from . import state as state
from .cuda_primitives import SourceModule
from .helpers import (
    _activate_context,
    _context_from_handle,
    _new_handle,
    _parse_kernel_params,
    _parse_local_size,
    _set_error,
    _to_bytes,
)
from .state import _CommandRecord, _ComputePlan


def stage_compute_plan_create(context, shader_source, bindings, pc_size, shader_name):
    ctx = _context_from_handle(int(context))
    if ctx is None:
        return 0

    source_bytes = _to_bytes(shader_source)
    shader_name_bytes = _to_bytes(shader_name)
    source_text = source_bytes.decode("utf-8", errors="replace")

    try:
        with _activate_context(ctx):
            module = SourceModule(
                source_text,
                no_extern_c=True,
                options=["-w"],
            )
            function = module.get_function("vkdispatch_main")
    except Exception as exc:
        _set_error(f"Failed to compile CUDA kernel '{shader_name_bytes.decode(errors='ignore')}': {exc}")
        return 0

    try:
        params = _parse_kernel_params(source_text)
        local_size = _parse_local_size(source_text)
    except Exception as exc:
        _set_error(f"Failed to parse CUDA kernel metadata: {exc}")
        return 0

    plan = _ComputePlan(
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

    return _new_handle(state._compute_plans, plan)


def stage_compute_plan_destroy(plan):
    if plan is None:
        return
    state._compute_plans.pop(int(plan), None)


def stage_compute_record(command_list, plan, descriptor_set, blocks_x, blocks_y, blocks_z):
    cl = state._command_lists.get(int(command_list))
    cp = state._compute_plans.get(int(plan))
    if cl is None or cp is None:
        _set_error("Invalid command list or compute plan handle for stage_compute_record")
        return

    cl.commands.append(
        _CommandRecord(
            plan_handle=int(plan),
            descriptor_set_handle=int(descriptor_set),
            blocks=(int(blocks_x), int(blocks_y), int(blocks_z)),
            pc_size=int(cp.pc_size),
        )
    )
