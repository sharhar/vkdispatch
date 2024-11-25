import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

def test_builder_basic():
    buff = vd.asbuffer(np.array([1, 2, 3, 4], dtype=np.float32))
    buff2 = vd.asbuffer(np.array([10, 20, 30, 40], dtype=np.float32))

    uniform_buffer = vd.Buffer((vd.get_context().uniform_buffer_alignment, ), vd.float32)

    my_builder = vc.ShaderBuilder()

    var_buff = my_builder.declare_buffer(vc.f32)
    var_buff2 = my_builder.declare_buffer(vc.f32)

    uniform_var = my_builder.declare_constant(vc.f32)

    var_buff[vc.global_invocation().x] += var_buff2[vc.global_invocation().x] - uniform_var

    shader_description = my_builder.build(4, 1, 1, "my_shader")

    print(shader_description.source)

    compute_plan = vd.ComputePlan(shader_description.source, shader_description.binding_type_list, shader_description.pc_size, shader_description.name)

    descriptor_set = vd.DescriptorSet(compute_plan._handle)

    descriptor_set.bind_buffer(uniform_buffer, 0, type=1)
    descriptor_set.bind_buffer(buff, var_buff.binding)
    descriptor_set.bind_buffer(buff2, var_buff2.binding)

    uniform_buffer_builder = vd.BufferBuilder(usage=vd.BufferUsage.UNIFORM_BUFFER)
    uniform_buffer_builder.register_struct("my_shader", shader_description.uniform_structure)
    uniform_buffer_builder.prepare(1)
    uniform_buffer_builder[("my_shader", shader_description.exec_count_name)] = [2, 1, 1, 0]
    uniform_buffer_builder[("my_shader", uniform_var.raw_name)] = 5

    uniform_buffer.write(uniform_buffer_builder.tobytes())

    cmd_list = vd.CommandList()

    cmd_list.record_compute_plan(compute_plan, descriptor_set, [1, 1, 1])

    cmd_list.submit(instance_count=1)

    assert np.allclose(buff.read(0), np.array([6, 17, 3, 4], dtype=np.float32))
