import vkdispatch as vd
import numpy as np

def compute_shader(*args, local_size: tuple[int, int, int] = None):
    my_local_size = local_size if local_size is not None else [vd.get_devices()[0].max_workgroup_size[0], 1, 1]

    def decorator(build_func):
        builder = vd.shader_builder()

        #pc_exec_count = vd.push_constant(vd.ivec3)

        pc_exec_count_var = builder.dynamic_constant(vd.uvec4, "exec_count") #builder.import_push_constant(pc_exec_count, "exec_count")
        
        builder.if_statement(pc_exec_count_var[0] <= builder.global_x)
        builder.return_statement()
        builder.end_if()

        func_args = []

        for buff in args:
            if isinstance(buff, vd.buffer):
                func_args.append(builder.static_buffer(buff))
            else:
                raise ValueError("Only buffers are supported as static arguments!")

        if len(func_args) > 0:
            build_func(builder, *func_args)
        else:
            build_func(builder)

        plan = vd.compute_plan(builder.build(my_local_size[0], my_local_size[1], my_local_size[2]), builder.binding_count, builder.pc_size)

        pc_buff = vd.push_constant_buffer(builder.pc_dict)

        for binding in builder.bindings:
            plan.bind_buffer(binding[0], binding[1])

        class shader_wrapper:
            def __init__(self):
                pass

            def __getitem__(self, exec_dims: tuple | int):
                my_blocks = [exec_dims, 1, 1]
                my_cmd_list = [None]

                if isinstance(exec_dims, tuple):
                    my_blocks = [1, 1, 1]

                    for i, val in enumerate(exec_dims):
                        if isinstance(val, int) or np.issubdtype(type(val), np.integer):
                            my_blocks[i] = val
                        else:
                            if not isinstance(val, vd.command_list):
                                raise ValueError(f"Invalid dimension '{val}'!")
                            
                            if not i == len(exec_dims) - 1:
                                raise ValueError("Only the last dimension can be a command list!")
                            
                            my_cmd_list[0] = val

                pc_buff["exec_count"] = [my_blocks[0], my_blocks[1], my_blocks[2], 0]
 

                my_blocks[0] = (my_blocks[0] + my_local_size[0] - 1) // my_local_size[0]
                my_blocks[1] = (my_blocks[1] + my_local_size[1] - 1) // my_local_size[1]
                my_blocks[2] = (my_blocks[2] + my_local_size[2] - 1) // my_local_size[2]

                def wrapper_func(**kwargs):
                    for key, val in kwargs.items():
                        pc_buff[key] = val

                    if len(kwargs) == len(builder.pc_dict) - 1:
                        if my_cmd_list[0] is None:
                            my_cmd_list[0] = vd.get_command_list()
                        
                        my_cmd_list[0].add_pc_buffer(pc_buff)
                        plan.record(my_cmd_list[0], my_blocks)
                        my_cmd_list[0].submit()
                    
                    if my_cmd_list[0] is None:
                        raise ValueError("Must provide all dynamic constants if no command list is specified!")

                    my_cmd_list[0].add_pc_buffer(pc_buff)
                    plan.record(my_cmd_list[0], my_blocks)

                    return pc_buff
                return wrapper_func
        
        wrapper = shader_wrapper()
        return wrapper
    return decorator