import vkdispatch as vd

from typing import Callable, Any

def dispatch(plan: 'vd.compute_plan', blocks: tuple[int, int, int]) -> None:
    command_list = vd.command_list()
    plan.record(command_list, blocks)
    command_list.submit()

def dispatch_shader(build_func: Callable[['vd.shader_builder', Any], None], blocks: tuple[int, int, int], local_size: tuple[int, int, int], static_args: list[vd.buffer | vd.image] = []) -> None:
    plan = vd.build_compute_plan(build_func, local_size, static_args)
    dispatch(plan, blocks)
    