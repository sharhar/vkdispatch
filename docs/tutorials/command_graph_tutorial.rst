Command Graph Recording
=======================

``CommandGraph`` is the high-level recording API in vkdispatch. It lets you queue
multiple shader dispatches and submit them together, with automatic descriptor/uniform
handling.

When to Use a CommandGraph
--------------------------

Use ``CommandGraph`` when you want:

* Multiple dispatches in one recorded sequence.
* Explicit control over when work is submitted.
* Lower overhead than immediate submit-per-call flows.

Single Graph, Multiple Dispatches
---------------------------------

.. code-block:: python

   import numpy as np
   import vkdispatch as vd
   import vkdispatch.codegen as vc
   from vkdispatch.codegen.abreviations import *

   graph = vd.CommandGraph()

   @vd.shader("buff.size")
   def add_scalar(buff: Buff[f32], value: Const[f32]):
       tid = vc.global_invocation_id().x
       buff[tid] = buff[tid] + value

   arr = np.arange(32, dtype=np.float32)
   buff = vd.asbuffer(arr)

   # Record 3 dispatches, then submit once.
   add_scalar(buff, 1.0, graph=graph)
   add_scalar(buff, 1.0, graph=graph)
   add_scalar(buff, 1.0, graph=graph)

   graph.submit()
   vd.queue_wait_idle()

   out = buff.read(0)
   print(np.allclose(out, arr + 3.0))  # True

Immediate vs Deferred Submission
--------------------------------

``CommandGraph`` supports two common modes:

* Deferred mode (default): record first, call ``submit()`` later.
* Immediate mode: ``submit_on_record=True`` to submit each record call.

.. code-block:: python

   immediate_graph = vd.CommandGraph(reset_on_submit=True, submit_on_record=True)

In practice, deferred mode is usually better for batching work and reducing submission
overhead.

Global Graphs and Thread-Local Behavior
---------------------------------------

vkdispatch keeps a thread-local default graph used when no explicit ``graph=...`` is
provided.

* ``vd.global_graph()`` returns the current graph for the thread.
* ``vd.default_graph()`` creates/returns the default immediate graph.
* ``vd.set_global_graph(graph)`` sets a custom graph for the current thread.

For reproducible behavior in larger programs, passing ``graph=...`` explicitly is
recommended.

CommandGraph API Reference
--------------------------

See the :doc:`Full Python API Reference <../python_api>` for complete API details on:

* ``vkdispatch.CommandGraph``
* ``vkdispatch.global_graph``
* ``vkdispatch.default_graph``
* ``vkdispatch.set_global_graph``
