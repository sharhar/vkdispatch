import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np
import threading
import time

def test_concurrent_shader_generation_robust():
    barrier_enter = threading.Barrier(2)
    barrier_exit = threading.Barrier(2)

    thread_data = {}
    thread_errors = []

    def thread_task(thread_id):
        try:
            unique_name = f"var_thread_{thread_id}"
            
            @vd.shader(exec_size=(1,))
            def concurrent_shader(buf: vc.Buff[vc.f32]):
                barrier_enter.wait()
                
                active_builder = vc.get_builder()
                thread_data[f"builder_{thread_id}"] = active_builder
                
                reg = vc.new_float_register(1.0, var_name=unique_name)
                buf[0] = reg

                barrier_exit.wait()

            concurrent_shader.build()
            
            thread_data[f"source_{thread_id}"] = concurrent_shader.source

        except Exception as e:
            thread_errors.append(e)
    
    t1 = threading.Thread(target=thread_task, args=(1,))
    t2 = threading.Thread(target=thread_task, args=(2,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    if thread_errors:
        raise RuntimeError(f"Thread failed: {thread_errors[0]}")
    
    b1 = thread_data["builder_1"]
    b2 = thread_data["builder_2"]
    
    assert b1 is not b2, (
        f"THREAD SAFETY FAILURE: Both threads retrieved the exact same "
        f"ShaderBuilder instance ({id(b1)}). This means `GlobalBuilder` is shared."
    )

    src_1 = thread_data["source_1"]
    src_2 = thread_data["source_2"]

    assert "var_thread_1" in src_1, "Thread 1 failed to generate its own variable."
    assert "var_thread_2" not in src_1, (
        "LEAK DETECTED: Thread 2's variable 'var_thread_2' appeared in Thread 1's source code."
    )

    assert "var_thread_2" in src_2, "Thread 2 failed to generate its own variable."
    assert "var_thread_1" not in src_2, (
        "LEAK DETECTED: Thread 1's variable 'var_thread_1' appeared in Thread 2's source code."
    )