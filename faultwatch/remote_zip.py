"""Read members of a large remote zip archive with HTTP range requests, so
one wind farm can be pulled out of a multi-GB archive without downloading it."""
from __future__ import annotations

import io
import urllib.request
import zipfile


class _HttpFile(io.RawIOBase):
    def __init__(self, url: str):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def RemoteZip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(_HttpFile(url), buffer_size=1 << 22))
