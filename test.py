import vkdispatch as vd
import tqdm

#vd.initialize(log_level=vd.LogLevel.INFO, debug_mode=True)

vd.make_context(multi_device=True, multi_queue=True)

buff = vd.Buffer((1024, 1024), vd.complex64)

cmd_stream = vd.CommandStream()

vd.fft.fft(buff, cmd_stream=cmd_stream)

for i in tqdm.tqdm(range(10000)):
    cmd_stream.submit_any(instance_count=100)

vd.queue_wait_idle()
