import subprocess
import sys
import textwrap



def test_opencl_codegen_scalarizes_packed_vec3_buffer_access():
    script = textwrap.dedent(
        """
        import vkdispatch as vd
        import vkdispatch.codegen as vc

        vd.initialize(backend="dummy")
        vd.set_dummy_context_params(
            subgroup_size=32,
            max_workgroup_size=(128, 1, 1),
            max_workgroup_count=(65535, 65535, 65535),
        )
        vc.set_codegen_backend("opencl")

        @vd.shader(4)
        def copy3(dst: vc.Buff[vc.uv3], src: vc.Buff[vc.uv3]):
            tid = vc.global_invocation_id().x
            dst[tid] = src[tid]

        print(copy3.get_src().code)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    src = result.stdout

    assert "buf0_scalar[" in src
    assert "buf1_scalar[" in src
    assert "buf0.data[" not in src
    assert "buf1.data[" not in src
    assert "* 3) + 0" in src
    assert "* 3) + 1" in src
    assert "* 3) + 2" in src
