Shader Authoring and Dispatch
=============================

vkdispatch lets you write compute logic in Python syntax and compile it to GLSL at
runtime. This page covers shader launch patterns and the key semantics of vkdispatch's
runtime shader generation model.

Examples below omit ``vd.initialize()`` and ``vd.make_context()`` because vkdispatch
creates them automatically on first runtime use. Call them manually only when you need
custom initialization/context settings.

Runtime Generation Model
------------------------

``@vd.shader`` executes your Python function with tracing objects and emits shader code
as each operation runs. In practice:

1. vkdispatch inspects type-annotated arguments and creates shader variables.
2. arithmetic, indexing, swizzles, and assignment append GLSL statements.
3. the generated source is compiled into a compute plan and then dispatched.

This is different from AST/IR compilers: it is a forward streaming model, so explicit
register materialization and explicit shader control-flow helpers matter for performance
and correctness.

Imports and Type Annotations
----------------------------

Most shader examples use these imports:

.. code-block:: python

   import vkdispatch as vd
   import vkdispatch.codegen as vc
   from vkdispatch.codegen.abreviations import *

* ``Buff[...]`` is a shader buffer argument type.
* ``Const[...]`` is a uniform/constant argument type.
* Dtype aliases such as ``f32``, ``i32``, and ``v2`` come from abbreviations.

Basic In-Place Kernel
---------------------

.. code-block:: python

   import numpy as np
   import vkdispatch as vd
   import vkdispatch.codegen as vc
   from vkdispatch.codegen.abreviations import *

   @vd.shader("buff.size")
   def add_scalar(buff: Buff[f32], bias: Const[f32]):
       tid = vc.global_invocation_id().x
       buff[tid] = buff[tid] + bias

   arr = np.arange(32, dtype=np.float32)
   buff = vd.asbuffer(arr)
   add_scalar(buff, 1.5)

   result = buff.read(0)
   print(result[:4])  # [1.5 2.5 3.5 4.5]

Launch Configuration
--------------------

Use one of these launch patterns:

* String expression (evaluated from function argument names):

  .. code-block:: python

     @vd.shader("in_buf.size")
     def kernel(in_buf: Buff[f32], out_buf: Buff[f32]):
         ...

* Fixed total dispatch size:

  .. code-block:: python

     @vd.shader(exec_size=(1024, 1, 1))
     def kernel(...):
         ...

* Dynamic size from call arguments:

  .. code-block:: python

     @vd.shader(exec_size=lambda args: args.in_buf.size)
     def kernel(in_buf: Buff[f32], out_buf: Buff[f32]):
         ...

* Explicit workgroups instead of ``exec_size``:

  .. code-block:: python

     @vd.shader(workgroups=(64, 1, 1), local_size=(128, 1, 1))
     def kernel(...):
         ...

``exec_size`` and ``workgroups`` are mutually exclusive.
The string form is often the most concise option for argument-dependent dispatch size.

You can also override launch parameters per call:

.. code-block:: python

   # Reuse the same compiled shader with different dispatch sizes.
   add_scalar(buff, 1.5, exec_size=buff.size)

Symbolic Expressions vs Mutable Registers
-----------------------------------------

vkdispatch variables are symbolic by default. Reusing an expression in multiple places
inlines that expression each time in generated code.

To materialize a value once and mutate it, convert it to a register with
``to_register()``:

.. code-block:: python

   @vd.shader("buff.size")
   def register_example(buff: Buff[f32]):
       tid = vc.global_invocation_id().x

       # Expression variable: may be inlined at each use.
       expr = vc.sin(tid * 0.1)

       # Register variable: emitted once, then reused.
       cached = expr.to_register("cached")

       buff[tid] = cached * 2.0 + cached / 3.0

Register Store Syntax (``[:]``)
-------------------------------

Python assignment rebinding (``x = ...``) changes the Python name, not the generated
shader register. To emit a GLSL assignment into an existing register, use full-slice
store syntax ``x[:] = ...``.

.. code-block:: python

   @vd.shader("buff.size")
   def register_store(buff: Buff[f32]):
       tid = vc.global_invocation_id().x
       value = buff[tid].to_register("value")
       value[:] = value * 0.5 + 1.0
       buff[tid] = value

Shader Control Flow vs Python Control Flow
------------------------------------------

Native Python control flow with vkdispatch variables is intentionally blocked:

.. code-block:: python

   @vd.shader("buff.size")
   def bad_branch(buff: Buff[f32]):
       tid = vc.global_invocation_id().x
       if tid < 10:  # Raises ValueError: vkdispatch variables are not Python booleans.
           buff[tid] = 1.0

Use shader control-flow helpers so both branches are emitted into generated code:

.. code-block:: python

   @vd.shader("buff.size")
   def threshold(buff: Buff[f32], cutoff: Const[f32]):
       tid = vc.global_invocation_id().x

       vc.if_statement(buff[tid] > cutoff)
       buff[tid] = 1.0
       vc.else_statement()
       buff[tid] = 0.0
       vc.end()

Generation-Time Specialization (Meta-Programming)
-------------------------------------------------

Because kernel bodies execute as normal Python during generation, Python loops and
conditionals are useful for specialization and unrolling.

.. code-block:: python

   def make_unrolled_sum(unroll: int):
       @vd.shader("dst.size")
       def unrolled_sum(src: Buff[f32], dst: Buff[f32]):
           tid = vc.global_invocation_id().x
           base = (tid * unroll).to_register("base")
           acc = vc.new_float_register(0.0)

           # Unrolled at generation time.
           for i in range(unroll):
               acc += src[base + i]

           dst[tid] = acc

       return unrolled_sum

   sum4 = make_unrolled_sum(4)
   sum8 = make_unrolled_sum(8)

   # sum4 and sum8 compile to different shaders with different unrolled bodies.

Mapping Functions
-----------------

Mapping functions are reusable typed snippets (often used with reductions and FFT I/O).

.. code-block:: python

   @vd.map
   def square_value(x: Buff[f32]) -> f32:
       idx = vd.reduce.mapped_io_index()
       return x[idx] * x[idx]

You can pass mapping functions into APIs that accept ``mapping_function``,
``input_map``, or ``output_map`` arguments.

Inspecting Generated Shader Source
----------------------------------

A built shader can be printed for debugging:

.. code-block:: python

   print(add_scalar)

This prints GLSL-like generated source with line numbers, which is useful when debugging
type issues or unsupported expressions.

Common Notes
------------

* All shader parameters must be type annotated.
* Buffer/image arguments must use codegen types (for example, ``Buff[f32]``, ``Img2[f32]``).
* If you need batched submissions, prefer :doc:`Command Graph Recording <command_graph_tutorial>`.

Shader API Reference
--------------------

See the :doc:`Full Python API Reference <../python_api>` for complete API details on:

* ``vkdispatch.shader``
* ``vkdispatch.map``
* ``vkdispatch.ShaderFunction``
* ``vkdispatch.MappingFunction``
