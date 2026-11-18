"""Pre-write space accounting for the owned LR diagnostic output tree.

Work cannot spend the closure reserve. Atomic replacements must fit alongside
the old file, and supervised log writers have disjoint, fixed-size slots.
Only one non-log writer is active at a time (worker, inner, observer, collector).
"""

import json
import io
import os
import resource
import stat
from contextlib import contextmanager
from pathlib import Path

CAP_BYTES = 2 * (1 << 30)
CLOSURE_RESERVE_BYTES = 1 << 20
FAILURE_RESERVE_BYTES = 4096
LOG_BYTES = 256 << 10  # per stream, not a per-file limit on numerical artifacts
LOG_PATHS = frozenset(Path(p) for p in (
    "supervisor_stdout.log", "supervisor_stderr.log",
    "attempt_001/worker_stdout.log", "attempt_001/worker_stderr.log"))


class ArtifactLimit(OSError):
    pass


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ArtifactSpace:
    def __init__(self, root, *, cap_bytes=CAP_BYTES, supervised=False):
        if type(cap_bytes) is not int or not 0 < cap_bytes <= CAP_BYTES:
            raise ValueError("invalid LR artifact cap")
        self.root = Path(root).resolve()
        self.cap_bytes = cap_bytes
        self.logs = LOG_PATHS if supervised else frozenset()

    def used_bytes(self, *, reserve_logs=False):
        used = len(self.logs) * LOG_BYTES if reserve_logs else 0

        def raise_walk_error(exc):
            raise exc

        for directory, dirs, files in os.walk(self.root, onerror=raise_walk_error):
            parent = Path(directory)
            if any((parent / sub).is_symlink() for sub in dirs):
                raise ArtifactLimit("lr_artifact_symlink_directory")
            for name in files:
                path = parent / name
                s = path.lstat()
                if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
                    raise ArtifactLimit(f"lr_artifact_not_owned_regular_file:{path.name}")
                if path.relative_to(self.root) in self.logs:
                    if s.st_size > LOG_BYTES:
                        raise ArtifactLimit(f"lr_artifact_log_slot_exceeded:{path.name}")
                    if reserve_logs:
                        continue  # safe even as this stream is still growing
                used += s.st_size
        return used

    def require(self, path, size, *, closing=False):
        path = Path(path).absolute()
        if not path.is_relative_to(self.root) or path.resolve() != path:
            raise ArtifactLimit(f"lr_artifact_path_not_writable:{path}")
        rel = path.relative_to(self.root)
        if rel in self.logs or type(size) is not int or size < 0:
            raise ArtifactLimit(f"lr_artifact_reservation_invalid:{rel}")
        reserve = FAILURE_RESERVE_BYTES if closing else CLOSURE_RESERVE_BYTES
        used = self.used_bytes(reserve_logs=True)
        if used + size > self.cap_bytes - reserve:
            raise ArtifactLimit(f"lr_artifact_cap_exceeded:{rel}:need={size}:"
                                f"available={max(0, self.cap_bytes - reserve - used)}")

    def write_bytes(self, path, blob, *, closing=False, append=False, exclusive=False, check=None):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp") if not append and not exclusive else path
        self.require(tmp, len(blob), closing=closing)
        if check is not None:
            check("lr_artifact_write:" + path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("ab" if append else "xb") as out:
            try:
                out.write(blob)
                out.flush()
                os.fsync(out.fileno())
            except OSError:
                if not append and not exclusive:
                    tmp.unlink()
                raise
        if tmp != path:
            os.replace(tmp, path)
        sync_directory(path.parent)

    def write_json(self, path, doc, *, closing=False, exclusive=False, check=None):
        self.write_bytes(path, (json.dumps(doc, indent=1, allow_nan=False) + "\n").encode(),
                         closing=closing, exclusive=exclusive, check=check)

    def write_closure_json(self, path, doc):
        self.write_json(path, doc, closing=True)

    def write_failure(self, path, reason):
        """Small exclusive fail-closed receipt, with its own reserved bytes."""
        path = Path(path)
        blob = (json.dumps({"state": "incomplete", "problem": str(reason)[:500]}) + "\n").encode()
        if len(blob) > FAILURE_RESERVE_BYTES or self.used_bytes(reserve_logs=True) + len(blob) > self.cap_bytes:
            raise ArtifactLimit("lr_artifact_failure_reserve_unavailable")
        if path.parent.resolve() != self.root:
            raise ArtifactLimit("lr_artifact_failure_outside_root")
        with path.open("xb") as out:
            out.write(blob)
            out.flush()
            os.fsync(out.fileno())
        sync_directory(path.parent)

    def write_npz(self, path, records, *, check):
        import numpy as np

        # np.savez is uncompressed; the exact ZIP (incl. headers) is produced
        # in memory. No on-disk byte is written before it fits the reservation.
        self.require(path, sum(a.nbytes for a in records.values()))
        check("lr_npz_serialize:" + path.name)
        with io.BytesIO() as stream:
            np.savez(stream, **records)
            with stream.getbuffer() as data:
                self.write_bytes(path, data, check=check)

    @contextmanager
    def checkpoint(self, path, modules, *, check):
        """Reserve an uncompressed tensor/metadata envelope, with a kernel backstop.

        Trainer.save keeps its native path/zip format and atomic rename. The
        limit applies even if the serializer's metadata exceeds the envelope.
        """
        path = Path(path)
        tensors = (v for module in modules for v in (*module.parameters(), *module.buffers()))
        bound = (256 << 10) + sum(v.untyped_storage().nbytes() + 4096 for v in tensors)
        self.require(path.with_name(path.name + ".tmp"), bound)
        check("lr_checkpoint_write:" + path.name)
        limits = resource.getrlimit(resource.RLIMIT_FSIZE)
        limit = bound if limits[0] == resource.RLIM_INFINITY else min(bound, limits[0])
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limits[1]))
        try:
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, limits)


def policy():
    return {"schema": "w1_d1_lr_artifact_space_v1", "cap_bytes": CAP_BYTES,
            "closure_reserve_bytes": CLOSURE_RESERVE_BYTES,
            "failure_reserve_bytes": FAILURE_RESERVE_BYTES,
            "log_bytes_per_stream": LOG_BYTES, "log_streams": len(LOG_PATHS)}
