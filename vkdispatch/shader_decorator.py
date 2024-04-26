import vkdispatch as vd

def compute_shader(*args, local_size: tuple[int, int, int] = None):
    my_local_size = local_size if local_size is not None else [vd.get_devices()[0].max_workgroup_size[0], 1, 1]

    def decorator(build_func):
        builder = vd.shader_builder()

        pc_exec_count = vd.push_constant(vd.ivec3)

        pc_exec_count_var = builder.import_push_constant(pc_exec_count, "exec_count")

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

        for binding in builder.bindings:
            plan.bind_buffer(binding[0], binding[1])

        class shader_wrapper:
            def __init__(self):
                pass

            def __getitem__(self, exec_dims: tuple[int, int, int] | int):
                my_blocks = exec_dims if isinstance(exec_dims, tuple) else [exec_dims, 1, 1]

                pc_exec_count.data[:3] = my_blocks

                my_blocks[0] = (my_blocks[0] + my_local_size[0] - 1) // my_local_size[0]
                my_blocks[1] = (my_blocks[1] + my_local_size[1] - 1) // my_local_size[1]
                my_blocks[2] = (my_blocks[2] + my_local_size[2] - 1) // my_local_size[2]

                def wrapper_func():
                    command_list = vd.command_list()
                    plan.record(command_list, my_blocks)
                    command_list.submit(pc_exec_count.data.tobytes(), 1)
                return wrapper_func
        
        wrapper = shader_wrapper()
        return wrapper
    return decorator