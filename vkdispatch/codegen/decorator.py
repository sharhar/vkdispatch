import vkdispatch as vd
import vkdispatch.codegen as vc

import functools
import numpy as np

import inspect

from typing import Callable

def shader(*args, local_size=None):
    def process_function(func):
        signature = inspect.signature(func)

        my_local_size = (
            local_size
            if local_size is not None
            else [vd.get_context().device_infos[0].max_workgroup_size[0], 1, 1]
        )

        vc.builder_obj.reset()

        pc_exec_count_var = vc.builder_obj.declare_constant(vd.uvec4, "exec_count")

        vc.if_statement(pc_exec_count_var.x <= vc.global_invocation.x)
        vc.return_statement()
        vc.end()

        func_args = []

        for param in signature.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")

            if not hasattr(param.annotation, '__args__'):
                raise TypeError(f"Argument '{param.name}: vd.{param.annotation.__name__}' must have a type annotation")

            if len(param.annotation.__args__) != 1:
                raise ValueError(f"Type '{param.name}: vd.{param.annotation.__name__}' must have exactly one type argument")

            type_arg: vd.dtype = param.annotation.__args__[0]

            if(issubclass(param.annotation.__origin__, vc.Buffer)):
                func_args.append(vc.builder_obj.declare_buffer(type_arg, var_name=f"__{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Image2D)):
                func_args.append(vc.builder_obj.declare_image(2, var_name=f"__{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Image3D)):
                func_args.append(vc.builder_obj.declare_image(3, var_name=f"__{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Constant)):
                func_args.append(vc.builder_obj.declare_constant(type_arg, var_name=f"__{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Variable)):
                func_args.append(vc.builder_obj.declare_variable(type_arg, var_name=f"__{param.name}"))
        
        func(*func_args)

        shader_source, pc_size, pc_dict, uniform_dict, binding_type_list = vc.builder_obj.build(
            my_local_size[0], my_local_size[1], my_local_size[2]
        )

        class Wrapper:
            def __init__(self, func):
                self.func = func

            def __repr__(self):
                return shader_source

            def __call__(self, *args, **kwargs):
                pass #self.func(*args, **kwargs)

        wrapper: str = Wrapper(func)

        return wrapper
    
    if len(args) == 1 and callable(args[0]):
        return process_function(args[0])
    return process_function