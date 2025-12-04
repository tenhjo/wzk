import hashlib


def hash_file(file):
    with open(file, "rb") as f:
        h = hashlib.file_digest(f, digest="sha256").digest()  # noqa: raising-bad-type
    h = int.from_bytes(h, byteorder="little")
    return h


def hash2(b):
    m = hashlib.sha256()
    m.update(b)
    h = m.digest()
    h = int.from_bytes(h, byteorder="little")
    return h


def test_hash2():
    import numpy as np
    np.random.seed(0)
    a = np.random.random(100000)

    print(hash2(a.tobytes()))
    print(hash2(a.tobytes()))

    print(__file__)
    print(hash_file(__file__))


if __name__ == "__main__":
    test_hash2()
