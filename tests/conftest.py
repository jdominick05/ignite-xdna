import gc
import time
import pytest

@pytest.fixture(autouse=True)
def cleanup_xrt_hardware_contexts():
    """Ensure PyXRT device handles are garbage collected and amdxe.sys slots recycled."""
    yield
    gc.collect()
    time.sleep(0.015)
