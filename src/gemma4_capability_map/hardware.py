from __future__ import annotations

import os
import platform

from .schemas import HardwareProfile


def detect_hardware_profile() -> HardwareProfile:
    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return HardwareProfile(
        platform=platform.system(),
        platform_version=platform.version(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        memory_gb=round(memory_bytes / (1024 ** 3), 2),
    )

