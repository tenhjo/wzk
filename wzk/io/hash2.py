from __future__ import annotations

import hashlib

from wzk.logger import setup_logger

logger = setup_logger(__name__)


def hash_file(file: str) -> int:  # TODO: pathlib — Path.read_bytes()
    with open(file, "rb") as f:
        h = hashlib.file_digest(f, digest="sha256").digest()
    h = int.from_bytes(h, byteorder="little")
    return h


def hash2(b: bytes) -> int:
    m = hashlib.sha256()
    m.update(b)
    h = m.digest()
    h = int.from_bytes(h, byteorder="little")
    return h


def test_hash2() -> None:
    import numpy as np

    np.random.seed(0)
    a = np.random.random(100000)

    logger.debug(hash2(a.tobytes()))
    logger.debug(hash2(a.tobytes()))

    logger.debug(__file__)
    logger.debug(hash_file(__file__))


if __name__ == "__main__":
    test_hash2()
