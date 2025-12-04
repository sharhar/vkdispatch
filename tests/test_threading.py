import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np
import threading
import time

def test_concurrent_shader_generation_robust():
    """
    Stresses the thread safety of the code generation engine.
    
    Uses double barriers to force two threads to be inside the active 
    'build' context simultaneously. 
    
    If state is shared (not thread-local):
    1. Both threads will report seeing the SAME builder object.
    2. Variables from Thread 2 will appear in Thread 1's source code.
    """
    
    # Barrier 1: Wait until both threads have started the build process 
    # and entered the python function. This ensures T2 has overwritten T1's global state.
    barrier_enter = threading.Barrier(2)
    
    # Barrier 2: Wait until both threads are done defining variables but BEFORE 
    # they return. This prevents T2 from restoring the global state while T1 is still working.
    barrier_exit = threading.Barrier(2)

    thread_data = {}
    thread_errors = []

    def thread_task(thread_id):
        try:
            # Unique marker to identify this thread's variables
            unique_name = f"var_thread_{thread_id}"
            
            @vd.shader(exec_size=(1,))
            def concurrent_shader(buf: vc.Buff[vc.f32]):
                # 1. Force Collision: Wait for the other thread to enter this function too.
                # If global state is shared, the last thread to enter (say T2) 
                # will have set the GlobalBuilder to T2's builder.
                barrier_enter.wait()
                
                # 2. Capture the 'active' builder seen by this thread.
                # In a broken implementation, T1 will see T2's builder here.
                active_builder = vc.get_builder()
                thread_data[f"builder_{thread_id}"] = active_builder
                
                # 3. Define a unique variable.
                # If broken, this registers into whichever builder is currently global.
                reg = vc.new_float_register(1.0, var_name=unique_name)
                buf[0] = reg

                # 4. Hold the lock: Do not let this thread exit (and restore the global builder)
                # until the other thread is also done defining its logic.
                barrier_exit.wait()

            # Trigger the execution of the python function
            concurrent_shader.build()
            
            # Save the final generated source code
            thread_data[f"source_{thread_id}"] = concurrent_shader.source

        except Exception as e:
            thread_errors.append(e)

    # --- Execution ---
    
    t1 = threading.Thread(target=thread_task, args=(1,))
    t2 = threading.Thread(target=thread_task, args=(2,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Rethrow any exceptions that happened inside threads
    if thread_errors:
        raise RuntimeError(f"Thread failed: {thread_errors[0]}")

    print(thread_data["source_1"])
    print(thread_data["source_2"])

    # --- Strict Assertions ---

    # 1. Object Identity Check
    # Even if source code looks okay by luck, the builder objects MUST be distinct instances.
    b1 = thread_data["builder_1"]
    b2 = thread_data["builder_2"]
    
    assert b1 is not b2, (
        f"THREAD SAFETY FAILURE: Both threads retrieved the exact same "
        f"ShaderBuilder instance ({id(b1)}). This means `GlobalBuilder` is shared."
    )

    # 2. Source Code Leakage Check
    src_1 = thread_data["source_1"]
    src_2 = thread_data["source_2"]

    # Thread 1 should ONLY have 'var_thread_1'
    assert "var_thread_1" in src_1, "Thread 1 failed to generate its own variable."
    assert "var_thread_2" not in src_1, (
        "LEAK DETECTED: Thread 2's variable 'var_thread_2' appeared in Thread 1's source code."
    )

    # Thread 2 should ONLY have 'var_thread_2'
    assert "var_thread_2" in src_2, "Thread 2 failed to generate its own variable."
    assert "var_thread_1" not in src_2, (
        "LEAK DETECTED: Thread 1's variable 'var_thread_1' appeared in Thread 2's source code."
    )

    print("Success: Threads maintained isolated builder contexts.")

if __name__ == "__main__":
    test_concurrent_shader_generation_robust()