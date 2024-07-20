import vkdispatch as vd
import vkdispatch.codegen as vc

import inspect

def shader(*args, local_size=None, workgroups=None, exec_size=None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def process_function(func):
        signature = inspect.signature(func)

        my_local_size = (
            local_size
            if local_size is not None
            else [vd.get_context().device_infos[0].max_workgroup_size[0], 1, 1]
        )

        vc.builder_obj.reset()

        vc.builder_obj.exec_count = vc.builder_obj.declare_constant(vd.uvec4, "exec_count")

        vc.if_statement(vc.builder_obj.exec_count.x <= vc.global_invocation.x)
        vc.return_statement()
        vc.end()

        func_args = []
        arg_names = []
        default_values = []

        for param in signature.parameters.values():
            if param.annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")

            if not hasattr(param.annotation, '__args__'):
                raise TypeError(f"Argument '{param.name}: vd.{param.annotation.__name__}' must have a type annotation")

            if len(param.annotation.__args__) != 1:
                raise ValueError(f"Type '{param.name}: vd.{param.annotation.__name__}' must have exactly one type argument")

            type_arg: vd.dtype = param.annotation.__args__[0]

            if(issubclass(param.annotation.__origin__, vc.Buffer)):
                func_args.append(vc.builder_obj.declare_buffer(type_arg)) #, var_name=f"{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Image2D)):
                func_args.append(vc.builder_obj.declare_image(2)) #, var_name=f"{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Image3D)):
                func_args.append(vc.builder_obj.declare_image(3)) #, var_name=f"{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Constant)):
                func_args.append(vc.builder_obj.declare_constant(type_arg, var_name=f"{param.name}"))
            elif(issubclass(param.annotation.__origin__, vc.Variable)):
                func_args.append(vc.builder_obj.declare_variable(type_arg, var_name=f"{param.name}"))

            arg_names.append(param.name)

            if param.default != inspect.Parameter.empty:
                default_values.append(param.default)
            else:
                default_values.append(None)
        
        func(*func_args)

        shader_source, pc_size, pc_dict, uniform_dict, binding_type_list = vc.builder_obj.build(
            my_local_size[0], my_local_size[1], my_local_size[2]
        )

        wrapper: str = vd.ShaderLauncher(
            shader_source,
            pc_size,
            pc_dict,
            uniform_dict,
            binding_type_list,
            list(zip(func_args, arg_names, default_values)),
            my_local_size,
            workgroups,
            exec_size
        )

        return wrapper
    
    if len(args) == 1 and callable(args[0]):
        return process_function(args[0])
    return process_function