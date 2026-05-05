def get_argument_names(fun):
    return fun.__code__.co_varnames[: fun.__code__.co_argcount]


def get_number_of_arguments(fun):
    return fun.__code__.co_argcount


def common_argument_wrapper(fun, **kwargs_common):
    def fun2(**kwargs):
        kwargs.update(kwargs_common)
        return fun(**kwargs)

    return fun2
