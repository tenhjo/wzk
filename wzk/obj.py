import copy


class CopyableObject(object):
    __slots__ = ()

    def copy(self):
        return copy.copy(self)


def obj2dict(o) -> dict:
    d = {}
    for attr in o.__slots__:
        d[attr] = getattr(o, attr)

    return d


def dict2obj(d: dict, o):
    for attr in d:
        setattr(o, attr, d[attr])

    return d
