"""Where the lab's data is kept off the controller: an S3-compatible bucket (Cloudflare R2) or a plain directory.
Keys are written once and never overwritten or deleted."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path


class Remote:
    def exists(self, key: str) -> bool: ...
    def put(self, key: str, path: Path) -> bool: ...          # False when the key was already there
    def get(self, key: str, path: Path) -> None: ...
    def keys(self, prefix: str) -> list[str]: ...


class DirRemote(Remote):
    """A directory standing in for a bucket (tests, an external drive)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()

    def put(self, key: str, path: Path) -> bool:
        dest = self.root / key
        if dest.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
        shutil.copyfile(path, tmp)
        try:
            os.link(tmp, dest)                  # atomic and refuses an existing name: write-once under races
        except FileExistsError:
            return False
        finally:
            os.unlink(tmp)
        return True

    def get(self, key: str, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / key, path)

    def keys(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*")
                      if p.is_file() and ".tmp-" not in p.name)


class S3Remote(Remote):
    """A bucket over the S3 API; R2 when LAB_STORE_ENDPOINT names its account endpoint. Credentials from the usual
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."""

    def __init__(self, bucket: str, endpoint: str | None = None, client=None):
        import boto3
        self.bucket = bucket
        self.s3 = client or boto3.client("s3", endpoint_url=endpoint, region_name="auto" if endpoint else None)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put(self, key: str, path: Path) -> bool:
        """A conditional put (If-None-Match: *): the bucket itself refuses a second write of a key."""
        from botocore.exceptions import ClientError
        try:
            with open(path, "rb") as fh:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=fh, IfNoneMatch="*")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                return False
            raise
        return True

    def get(self, key: str, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, str(path))

    def keys(self, prefix: str) -> list[str]:
        out = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            out += [o["Key"] for o in page.get("Contents", [])]
        return sorted(out)


class NoRemote(RuntimeError):
    pass


def open_remote(url: str | None = None) -> Remote:
    """`s3://bucket` or a directory path; default LAB_STORE (unset raises)."""
    url = url or os.environ.get("LAB_STORE")
    if not url:
        raise NoRemote("no store configured: set LAB_STORE to s3://<bucket> (with LAB_STORE_ENDPOINT for R2) or a directory")
    if url.startswith("s3://"):
        return S3Remote(url[5:].strip("/"), os.environ.get("LAB_STORE_ENDPOINT"))
    return DirRemote(Path(url.removeprefix("file://")).expanduser())
