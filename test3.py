from browser import document, window
import sys
import traceback

import vkdispatch as vd
import vkdispatch.base.context as vd_context
import vkdispatch.base.init as vd_init
import vkdispatch.execution_pipeline.command_graph as vd_command_graph
import vkdispatch.fft.shader_factories as vd_fft_shader_factories
import vkdispatch.codegen as vc


class OutputBuffer:
    def __init__(self):
        self._parts = []

    def write(self, value):
        if value is None:
            return
        self._parts.append(str(value))

    def flush(self):
        pass

    def get_text(self):
        return "".join(self._parts)


def _parse_positive_int(element_id, field_name):
    raw = document[element_id].value.strip()

    if raw == "":
        raise ValueError(f"{field_name} cannot be empty.")

    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")

    return parsed


def _read_device_options():
    return {
        "subgroup_size": _parse_positive_int("opt-subgroup-size", "Subgroup Size"),
        "max_workgroup_size": (
            _parse_positive_int("opt-wg-size-x", "Max Workgroup Size X"),
            _parse_positive_int("opt-wg-size-y", "Max Workgroup Size Y"),
            _parse_positive_int("opt-wg-size-z", "Max Workgroup Size Z"),
        ),
        "max_workgroup_invocations": _parse_positive_int(
            "opt-wg-invocations",
            "Max Workgroup Invocations",
        ),
        "max_workgroup_count": (
            _parse_positive_int("opt-wg-count-x", "Max Workgroup Count X"),
            _parse_positive_int("opt-wg-count-y", "Max Workgroup Count Y"),
            _parse_positive_int("opt-wg-count-z", "Max Workgroup Count Z"),
        ),
        "max_compute_shared_memory_size": _parse_positive_int(
            "opt-shared-memory",
            "Max Shared Memory (bytes)",
        ),
    }


def _reset_vkdispatch_runtime():
    context = getattr(vd_context, "__context", None)
    if context is not None:
        vd_context.destroy_context()

    vd_init.__initilized_instance = False
    vd_init.__device_infos = None

    state = vd_command_graph._global_graph
    for attr_name in ("custom_graph", "default_graph"):
        if hasattr(state, attr_name):
            delattr(state, attr_name)


def run_code(event):
    code = window.cmCode.getValue()
    window.cmOutput.setValue("")

    stdout_buffer = OutputBuffer()
    stderr_buffer = OutputBuffer()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buffer, stderr_buffer
    namespace = {"__name__": "__main__"}

    try:
        options = _read_device_options()
        _reset_vkdispatch_runtime()

        vd.initialize(backend="dummy")
        vd.get_context()
        vd.set_dummy_context_params(
            subgroup_size=options["subgroup_size"],
            max_workgroup_size=options["max_workgroup_size"],
            max_workgroup_invocations=options["max_workgroup_invocations"],
            max_workgroup_count=options["max_workgroup_count"],
            max_shared_memory=options["max_compute_shared_memory_size"],
        )

        # Set codegen backend based on toggle state
        backend = str(window.currentBackend)
        vc.set_codegen_backend(backend)
        vd_fft_shader_factories.cache_clear()

        exec(code, namespace)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        window.cmOutput.setValue(stdout_buffer.get_text() + stderr_buffer.get_text())


document["run-btn"].bind("click", run_code)

# Auto-run once when the Brython runtime is ready.
run_code(None)