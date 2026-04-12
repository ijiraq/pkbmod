"""In-flight RAM reservation for parallel Butler patch loads.

Process RSS alone does not cap peak use when several large loads run concurrently.
This module tracks a conservative *reserved* byte estimate (scaled by ref count)
so admission stays within a configurable fraction of the container cap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None


def stack_inputs_array_nbytes(stack_inputs: dict) -> int:
    """Sum nbytes of packed ndarray fields in ``stack_inputs`` (main heap cost)."""
    total = 0
    for key in ("datas", "masks", "variances", "psfs", "dmjds", "fwhms", "im_nums"):
        v = stack_inputs.get(key)
        if v is None:
            continue
        if hasattr(v, "nbytes"):
            total += int(v.nbytes)
        elif isinstance(v, (list, tuple)):
            for x in v:
                total += int(getattr(x, "nbytes", 0))
    return total


@dataclass
class LoadMemoryBudget:
    """Track reserved bytes for loads in progress and calibrate bytes per exposure."""

    container_cap_bytes: int
    max_ram_fraction: float
    bytes_per_exposure: float
    ema_alpha: float = 0.15
    reserved_bytes: float = 0.0

    def rss_bytes(self) -> int:
        if psutil is None:
            return 0
        return int(psutil.Process(os.getpid()).memory_info().rss)

    def memory_percent(self) -> float:
        return 100.0 * self.rss_bytes() / max(self.container_cap_bytes, 1)

    def estimate_patch_bytes(self, n_refs: int) -> float:
        return float(max(1, n_refs)) * self.bytes_per_exposure

    def reserve(self, estimate_bytes: float) -> None:
        self.reserved_bytes += estimate_bytes

    def release(self, estimate_bytes: float) -> None:
        self.reserved_bytes = max(0.0, self.reserved_bytes - estimate_bytes)

    def observe_completed_load(self, n_refs: int, array_nbytes: int) -> None:
        if n_refs <= 0 or array_nbytes <= 0:
            return
        observed = float(array_nbytes) / float(n_refs)
        a = self.ema_alpha
        self.bytes_per_exposure = (1.0 - a) * self.bytes_per_exposure + a * observed
        logger.debug(
            "LoadMemoryBudget: bytes_per_exposure now %.2f MiB "
            "(observed %.2f MiB/ref, n_refs=%d)",
            self.bytes_per_exposure / (1024**2),
            observed / (1024**2),
            n_refs,
        )

    def can_start_load(
        self,
        estimate_bytes: float,
        *,
        n_running_loads: int,
        max_parallel: int,
    ) -> bool:
        if n_running_loads >= max_parallel:
            return False
        cap = self.container_cap_bytes
        lim = cap * self.max_ram_fraction
        # Sum of in-flight reservations must stay under the same cap as RSS.
        if self.reserved_bytes + estimate_bytes > lim:
            # One patch can exceed the fraction (many refs); avoid deadlock when queue head
            # is huge but nothing is running yet — still gate on current RSS when available.
            if self.reserved_bytes == 0.0 and n_running_loads == 0:
                logger.warning(
                    "Patch load estimate (%.2f GiB) exceeds budget (%.2f GiB × %.0f%% = %.2f GiB); "
                    "starting anyway if RSS is below cap — risk of OOM; tune "
                    "--bytes-per-exposure-mib / --max-ram-percent / --container-ram-gib.",
                    estimate_bytes / (1024**3),
                    cap / (1024**3),
                    self.max_ram_fraction * 100,
                    lim / (1024**3),
                )
                if psutil is None:
                    return True
                return self.rss_bytes() <= lim
            return False
        if psutil is None:
            return True
        if self.rss_bytes() > lim:
            return False
        return True
