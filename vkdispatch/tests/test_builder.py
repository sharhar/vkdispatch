import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

def test_builder_basic():
    buff = vd.asbuffer(np.array([1, 2, 3, 4], dtype=np.float32))
    buff2 = vd.asbuffer(np.array([10, 20, 30, 40], dtype=np.float32))

    my_builder = vc.ShaderBuilder()

    var_buff = my_builder.declare_buffer(vc.f32)
    var_buff2 = my_builder.declare_buffer(vc.f32)

    uniform_var = my_builder.declare_constant(vc.f32)

    var_buff[vc.global_invocation.x] += var_buff2[vc.global_invocation.x] - uniform_var

    shader_source, pc_size, pc_dict, uniform_dict, binding_type_list = my_builder.build(4, 1, 1, "my_shader")

    compute_plan = vd.ComputePlan(shader_source, binding_type_list, pc_size, "add_buffers")

    descriptor_set = vd.DescriptorSet(compute_plan._handle)

    descriptor_set.bind_buffer(buff, var_buff.binding)
    descriptor_set.bind_buffer(buff2, var_buff2.binding)

    cmd_list = vd.CommandList()

    static_buffer_proxy = vc.BufferStructureProxy(uniform_dict, vd.get_context().uniform_buffer_alignment)

    static_buffer_proxy[uniform_var.raw_name] = 5

    cmd_list.add_desctiptor_set_and_static_constants(descriptor_set, static_buffer_proxy)

    compute_plan.record(cmd_list, descriptor_set, (1, 1, 1))

    cmd_list.submit()

    assert np.allclose(buff.read(0), np.array([6, 17, 28, 39], dtype=np.float32))





