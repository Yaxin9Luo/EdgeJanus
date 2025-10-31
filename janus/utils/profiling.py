import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _cuda_sync() -> None:
    if _cuda_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _gpu_mem_stats() -> Dict[str, Optional[int]]:
    if not _cuda_available():
        return {
            "allocated_bytes": None,
            "max_allocated_bytes": None,
            "reserved_bytes": None,
        }
    try:
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
    except Exception:
        return {
            "allocated_bytes": None,
            "max_allocated_bytes": None,
            "reserved_bytes": None,
        }


def _cpu_rss_bytes() -> Optional[int]:
    # Prefer psutil if available for accuracy; fallback to resource.
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss)
    except Exception:
        try:
            import resource  # type: ignore

            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is in kilobytes on Linux, bytes on macOS
            rss = usage.ru_maxrss
            # Heuristically convert KB to bytes if value is small
            if rss < 1 << 25:  # < 32MB likely bytes already
                return int(rss)
            return int(rss * 1024)
        except Exception:
            return None


def _gpu_env() -> Dict[str, Optional[str]]:
    if not _cuda_available():
        return {"device": "cpu", "gpu_name": None}
    try:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
    except Exception:
        name = None
    return {"device": "cuda", "gpu_name": name}


class JSONProfiler:
    """
    Minimal, low-overhead JSON profiler.

    Usage:
      prof = JSONProfiler(enabled=True, script="inference.py", model_path=model_path, out_path="metrics.json")
      prof.start_run(tag="setup")
      with prof.measure("model_load", reset_cuda_peak=True):
          ... load ...
      prof.end_run()

      prof.start_run(tag="inference")
      with prof.measure("generate", reset_cuda_peak=True):
          ...
      prof.end_run()

      prof.dump()
    """

    def __init__(self, enabled: bool, script: str, model_path: str, out_path: str = "metrics.json") -> None:
        self.enabled = bool(enabled)
        self.out_path = out_path
        self.report: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": script,
            "model_path": model_path,
            "env": _gpu_env(),
            "runs": [],  # type: ignore
        }
        self._current_run: Optional[Dict[str, Any]] = None

    def start_run(self, tag: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self._current_run = {"tag": tag, "sections": []}
        if extra:
            self._current_run.update(extra)

    def end_run(self) -> None:
        if not self.enabled:
            return
        if self._current_run is not None:
            self.report["runs"].append(self._current_run)  # type: ignore
        self._current_run = None

    @contextmanager
    def measure(self, name: str, reset_cuda_peak: bool = False):
        if not self.enabled:
            yield
            return
        if reset_cuda_peak and _cuda_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            t1 = time.perf_counter()
            section = {
                "name": name,
                "time_s": round(t1 - t0, 6),
                "gpu": _gpu_mem_stats(),
                "cpu_rss_bytes": _cpu_rss_bytes(),
            }
            if self._current_run is not None:
                self._current_run.setdefault("sections", []).append(section)

    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Append custom metrics to the most recent section in the current run."""
        if not self.enabled:
            return
        try:
            if self._current_run is None:
                return
            sections: List[Dict[str, Any]] = self._current_run.get("sections", [])  # type: ignore
            if not sections:
                return
            sections[-1].update(metrics)
        except Exception:
            pass

    def dump(self) -> None:
        if not self.enabled:
            return
        try:
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump(self.report, f, indent=2)
        except Exception:
            # Silently ignore write failures to avoid impacting inference.
            pass
