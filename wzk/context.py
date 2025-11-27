from contextlib import contextmanager


def __set_attribute_safe(obj, attribute, value, fail_on_error=True):
    try:
        setattr(obj, attribute, value)
    except AttributeError:
        if fail_on_error:
            raise AttributeError


@contextmanager
def set_attribute(obj, attribute, value, fail_on_error=True):

    old_value = getattr(obj, attribute)
    __set_attribute_safe(obj=obj, attribute=attribute, value=value, fail_on_error=fail_on_error)
    try:
        yield None

    finally:
        __set_attribute_safe(obj=obj, attribute=attribute, value=old_value, fail_on_error=fail_on_error)
