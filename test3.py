import vkdispatch as vd
import vkdispatch.codegen as vc

@vd.shader("buff.size")
def test_shader(buff: vc.Buff[vc.v2]):
    a = buff[vc.global_invocation().x] + 6

    b = vc.sin(a).copy()

    #buff[9].x = 5

    #buff[vc.global_invocation().x] += 10

buff = vd.Buffer((10, ), vc.v2)

test_shader(buff)

buff.read()