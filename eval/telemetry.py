import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from http.client import IncompleteRead
from threading import Lock
from unittest.mock import patch
from urllib.error import URLError

from scoring.engine import API_URL


ACTIVE = ContextVar("eval_usage", default=None)
LOCK = Lock()


@contextmanager
def track_usage():
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "missing_usage": 0}
    token = ACTIVE.set(usage)
    started = time.perf_counter()
    try:
        yield usage
    finally:
        usage["seconds"] = round(time.perf_counter() - started, 4)
        ACTIVE.reset(token)


@contextmanager
def capture_network_usage():
    # 透传评测调用上下文到冻结版内部的双评线程，避免并行时费用串记。
    network = urllib.request.urlopen
    submit = ThreadPoolExecutor.submit

    def submit_with_context(executor, fn, *args, **kwargs):
        return submit(executor, copy_context().run, fn, *args, **kwargs)

    def monitored(req, *args, **kwargs):
        state = ACTIVE.get()
        if state is None or getattr(req, "full_url", None) != API_URL:
            return network(req, *args, **kwargs)
        with LOCK:
            state["requests"] += 1
        try:
            with network(req, *args, **kwargs) as response:
                body = response.read()
        except IncompleteRead as exc:
            with LOCK:
                state["missing_usage"] += 1
            raise URLError("模型服务提前断开，HTTP 响应未读完整") from exc
        except OSError:
            with LOCK:
                state["missing_usage"] += 1
            raise
        try:
            usage = json.loads(body).get("usage")
        except (ValueError, AttributeError):
            usage = None
        with LOCK:
            if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
                state["input_tokens"] += usage["prompt_tokens"]
                state["output_tokens"] += usage["completion_tokens"]
            else:
                state["missing_usage"] += 1
        return io.BytesIO(body)

    with patch.object(urllib.request, "urlopen", monitored), patch.object(ThreadPoolExecutor, "submit", submit_with_context):
        yield
