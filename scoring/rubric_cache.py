import errno
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


# 预制表的结构或分值编译逻辑变化时递增。在线寻证和反馈修改无需重建表。
RUBRIC_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "预制与实测" / "评分细则缓存"


def save_json(path, value):
    path = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@contextmanager
def _build_lock(path):
    with path.open("a+b") as handle:
        deadline = time.monotonic() + 900
        if os.name == "nt":
            import msvcrt

            def acquire():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"等待评分细则预制超时：{path.name}") from exc
                time.sleep(0.2)
        yield


def load_or_build_rubric(cache_dir, cache_input, build, *, engine="v0"):
    encoded = json.dumps([engine, RUBRIC_VERSION, cache_input], ensure_ascii=False, sort_keys=True).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    with _build_lock(cache_dir / f"{key}.lock"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        rubric = build()
        save_json(path, rubric)
        return rubric
