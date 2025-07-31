import vkdispatch as vd
import vkdispatch.codegen as vc



@vd.shader()
def demo_shader(buff: vc.Buff[vc.f32]):
    tid = vc.global_invocation().x
    
    vc.if_statement(tid < 10)
    buff[tid] = 1.0
    vc.else_statement()    
    buff[tid] = 0.0
    vc.end()

print(demo_shader)