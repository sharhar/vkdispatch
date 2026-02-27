from __future__ import annotations

from typing import List, Optional, Tuple

from . import state as state
from .helpers import (
    activate_context,
    build_kernel_args_template,
    estimate_kernel_param_size_bytes,
    new_handle,
    queue_indices,
    set_error,
    stream_for_queue,
    to_bytes,
)
from .state import CUDACommandList, CUDAComputePlan, CUDACommandRecord

from .descriptor_sets import CUDADescriptorSet

import dataclasses

@dataclasses.dataclass
class CUDAResolvedLaunch:
    plan: CUDAComputePlan
    blocks: Tuple[int, int, int]
    descriptor_set: Optional[CUDADescriptorSet]
    pc_size: int
    pc_offset: int
    static_args: Optional[Tuple[object, ...]] = None

def command_list_create(context):
    if int(context) not in state.contexts:
        set_error("Invalid context handle for command_list_create")
        return 0

    return new_handle(state.command_lists, CUDACommandList(context_handle=int(context)))


def command_list_destroy(command_list):
    obj = state.command_lists.pop(int(command_list), None)
    if obj is None:
        return

    ctx = state.contexts.get(obj.context_handle)
    if ctx is None:
        return


def command_list_get_instance_size(command_list):
    obj = state.command_lists.get(int(command_list))
    if obj is None:
        return 0

    return int(sum(int(command.pc_size) for command in obj.commands))


def command_list_reset(command_list):
    obj = state.command_lists.get(int(command_list))
    if obj is None:
        return

    obj.commands = []


def command_list_submit(command_list, data, instance_count, index):
    obj = state.command_lists.get(int(command_list))
    if obj is None:
        return True

    ctx = state.contexts.get(obj.context_handle)
    if ctx is None:
        set_error(f"Missing context for command list {command_list}")
        return True

    instance_count = int(instance_count)
    if instance_count <= 0:
        return True

    instance_size = command_list_get_instance_size(command_list)
    payload = to_bytes(data)
    expected_payload_size = int(instance_size) * int(instance_count)

    if expected_payload_size == 0:
        if len(payload) != 0:
            set_error(
                f"Unexpected push-constant data for command list with instance_size=0 "
                f"(got {len(payload)} bytes)."
            )
            return True
    elif len(payload) != expected_payload_size:
        set_error(
            f"Push-constant data size mismatch. Expected {expected_payload_size} bytes "
            f"(instance_size={instance_size}, instance_count={instance_count}) but got {len(payload)} bytes."
        )
        return True

    queue_targets = queue_indices(ctx, int(index), all_on_negative=True)
    if len(queue_targets) == 0:
        queue_targets = [0]

    try:
        with activate_context(ctx):
            for queue_index in queue_targets:
                stream = stream_for_queue(ctx, queue_index)
                resolved_launches: List[CUDAResolvedLaunch] = []
                per_instance_offset = 0

                for command in obj.commands:
                    plan = state.compute_plans.get(command.plan_handle)
                    if plan is None:
                        raise RuntimeError(f"Invalid compute plan handle {command.plan_handle}")

                    descriptor_set = None
                    if command.descriptor_set_handle != 0:
                        descriptor_set = CUDADescriptorSet.from_handle(command.descriptor_set_handle)
                        if descriptor_set is None:
                            raise RuntimeError(
                                f"Invalid descriptor set handle {command.descriptor_set_handle}"
                            )

                    command_pc_size = int(command.pc_size)
                    first_instance_payload = b""
                    if command_pc_size > 0 and len(payload) > 0:
                        first_instance_payload = payload[per_instance_offset: per_instance_offset + command_pc_size]

                    static_args = None
                    if command_pc_size == 0:
                        static_args = build_kernel_args_template(plan, descriptor_set, b"")
                        size_check_args = static_args
                    else:
                        size_check_args = build_kernel_args_template(
                            plan,
                            descriptor_set,
                            first_instance_payload,
                        )

                    estimated_param_size = estimate_kernel_param_size_bytes(size_check_args)
                    if estimated_param_size > int(ctx.max_kernel_param_size):
                        shader_name = plan.shader_name.decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Kernel '{shader_name}' launch parameters require "
                            f"{estimated_param_size} bytes, exceeding device limit "
                            f"{ctx.max_kernel_param_size} bytes. "
                            "Reduce by-value uniform/push-constant payload size or switch large "
                            "uniform data to buffer-backed arguments."
                        )
                    resolved_launches.append(
                        CUDAResolvedLaunch(
                            plan=plan,
                            blocks=command.blocks,
                            descriptor_set=descriptor_set,
                            pc_size=command_pc_size,
                            pc_offset=per_instance_offset,
                            static_args=static_args,
                        )
                    )
                    per_instance_offset += command_pc_size

                if per_instance_offset != instance_size:
                    raise RuntimeError(
                        f"Internal command list size mismatch: computed {per_instance_offset} bytes, "
                        f"expected {instance_size} bytes."
                    )

                for instance_index in range(instance_count):
                    instance_base_offset = instance_index * instance_size
                    for launch in resolved_launches:
                        if launch.static_args is not None:
                            args = launch.static_args
                        else:
                            pc_start = instance_base_offset + launch.pc_offset
                            pc_end = pc_start + launch.pc_size
                            pc_payload = payload[pc_start:pc_end]
                            args = build_kernel_args_template(
                                launch.plan,
                                launch.descriptor_set,
                                pc_payload,
                            )

                        launch.plan.function(
                            *args,
                            block=launch.plan.local_size,
                            grid=launch.blocks,
                            stream=stream,
                        )
    except Exception as exc:
        set_error(f"Failed to submit CUDA command list: {exc}")

    return True

def stage_compute_record(command_list, plan, descriptor_set, blocks_x, blocks_y, blocks_z):
    cl = state.command_lists.get(int(command_list))
    cp = state.compute_plans.get(int(plan))
    if cl is None or cp is None:
        set_error("Invalid command list or compute plan handle for stage_compute_record")
        return

    cl.commands.append(
        CUDACommandRecord(
            plan_handle=int(plan),
            descriptor_set_handle=int(descriptor_set),
            blocks=(int(blocks_x), int(blocks_y), int(blocks_z)),
            pc_size=int(cp.pc_size),
        )
    )
