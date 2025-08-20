import vkdispatch as vd
import tqdm

#vd.initialize(log_level=vd.LogLevel.INFO, debug_mode=True)

vd.make_context(multi_device=True, multi_queue=True)

buff = vd.Buffer((1024, 1024), vd.complex64)

graph = vd.CommandGraph()

vd.fft.fft(buff, graph=graph)

for i in tqdm.tqdm(range(10000)):
    graph.submit_any(instance_count=100)

vd.queue_wait_idle()
