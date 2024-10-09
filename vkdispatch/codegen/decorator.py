import vkdispatch as vd
import vkdispatch.codegen as vc

import inspect

def shader(*args, local_size=None, workgroups=None, exec_size=None, signature: tuple = None):
    if workgroups is not None and exec_size is not None:
        raise ValueError("Cannot specify both 'workgroups' and 'exec_size'")

    def process_function(func):
        func_signature = inspect.signature(func)

        my_local_size = (
            local_size
            if local_size is not None
            else [vd.get_context().max_workgroup_size[0], 1, 1]
        )

        vc.builder_obj.reset()

        vc.builder_obj.exec_count = vc.builder_obj.declare_constant(vd.uvec4)

        vc.if_statement(vc.builder_obj.exec_count.x <= vc.global_invocation.x)
        vc.return_statement()
        vc.end()

        func_args = []
        arg_names = []
        default_values = []
        args_dict = {}

        my_type_annotations = (
            [(
                param.annotation, param.name, 
                param.default if param.default is not inspect.Parameter.empty else None
            ) for param in func_signature.parameters.values()]
            if signature is None else 
            [(
                sign, f"param{ii}", None
            ) for ii, sign in enumerate(signature)]
        )

        for my_annotation, my_name, my_default in my_type_annotations:

            if my_annotation == inspect.Parameter.empty:
                raise ValueError("All parameters must be annotated")

            if not hasattr(my_annotation, '__args__'):
                raise TypeError(f"Argument '{my_name}: vd.{my_annotation}' must have a type annotation")
            
            if(issubclass(my_annotation.__origin__, vc.ConstantArray)):
                if len(my_annotation.__args__) != 2:
                    raise ValueError(f"Type '{my_name}: vd.{my_annotation.__name__}' must have exactly two type arguments")
                
                type_arg: vd.dtype = my_annotation.__args__[0]
                
                new_constant = vc.builder_obj.declare_constant(type_arg, my_annotation.__args__[1])
                args_dict[new_constant.name] = my_name
                func_args.append(new_constant)
            elif(issubclass(my_annotation.__origin__, vc.VariableArray)):
                if len(my_annotation.__args__) != 2:
                    raise ValueError(f"Type '{my_name}: vd.{my_annotation.__name__}' must have exactly two type arguments")
                
                type_arg: vd.dtype = my_annotation.__args__[0]
                new_constant = vc.builder_obj.declare_variable(type_arg, my_annotation.__args__[1])
                args_dict[new_constant.name] = my_name
                func_args.append(new_constant)
            else:
                if len(my_annotation.__args__) != 1:
                    raise ValueError(f"Type '{my_name}: vd.{my_annotation.__name__}' must have exactly one type argument")

                type_arg: vd.dtype = my_annotation.__args__[0]

                if(issubclass(my_annotation.__origin__, vc.Buffer)):
                    func_args.append(vc.builder_obj.declare_buffer(type_arg))
                elif(issubclass(my_annotation.__origin__, vc.Image1D)):
                    func_args.append(vc.builder_obj.declare_image(1))
                elif(issubclass(my_annotation.__origin__, vc.Image2D)):
                    func_args.append(vc.builder_obj.declare_image(2))
                elif(issubclass(my_annotation.__origin__, vc.Image3D)):
                    func_args.append(vc.builder_obj.declare_image(3))
                elif(issubclass(my_annotation.__origin__, vc.Constant)):
                    new_constant = vc.builder_obj.declare_constant(type_arg)
                    args_dict[new_constant.name] = my_name
                    func_args.append(new_constant)
                elif(issubclass(my_annotation.__origin__, vc.Variable)):
                    new_variable = vc.builder_obj.declare_variable(type_arg)
                    args_dict[new_variable.name] = my_name
                    func_args.append(new_variable)

            arg_names.append(my_name)

            if my_default is not inspect.Parameter.empty:
                default_values.append(my_default)
            else:
                default_values.append(None)
        
        func(*func_args)

        shader_source, pc_size, pc_dict, uniform_dict, binding_type_list = vc.builder_obj.build(
            my_local_size[0], my_local_size[1], my_local_size[2]
        )

        wrapper: str = vd.ShaderLauncher(
            shader_source,
            func.__name__,
            pc_size,
            pc_dict,
            uniform_dict,
            binding_type_list,
            list(zip(func_args, arg_names, default_values)),
            my_local_size,
            workgroups,
            exec_size,
            args_dict
        )  # type: ignore

        return wrapper
    
    if len(args) == 1 and callable(args[0]):
        return process_function(args[0])
    return process_function