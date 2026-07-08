import atexit
import functools

import torch

from ..bindings.internal import runtime as bindings

HOSTFUNC_USER_DATA_HANDLES = set()


def launch_hostfunc(hostfunc, *args, **kwargs):
    stream = torch.cuda.current_stream()
    is_capturing = torch.cuda.is_current_stream_capturing()
    handle = bindings.launch_hostfunc(stream.cuda_stream, not is_capturing,
                                      hostfunc, *args, **kwargs)
    if is_capturing:
        HOSTFUNC_USER_DATA_HANDLES.add(handle)
    else:
        assert handle is None
    return handle


def hostfunc(fn=None, *, on_error=None):
    """Decorate `fn` to launch as a CUDA host callback on the current stream.

    An exception escaping a host callback cannot propagate to the launching
    thread: the CUDA trampoline logs and swallows it, so the failure is
    silently lost and, under CUDA graph capture, leaves the captured node in
    an undefined state. Callbacks that need to observe failures must pass
    `on_error`, called as `on_error(exc, fn, args, kwargs)` on the callback
    thread; it must not raise or block.
    """
    if on_error is not None and not callable(on_error):
        raise TypeError("hostfunc(...): on_error must be callable or None.")

    def decorate(fn):
        callback = fn
        if on_error is not None:

            @functools.wraps(fn)
            def callback(*args, **kwargs):
                try:
                    fn(*args, **kwargs)
                except BaseException as e:
                    on_error(e, fn, args, kwargs)

        def wrapper(*args, **kwargs):
            return launch_hostfunc(callback, *args, **kwargs)

        return wrapper

    return decorate if fn is None else decorate(fn)


def free_hostfunc_user_data(handle: int):
    if handle not in HOSTFUNC_USER_DATA_HANDLES:
        raise ValueError(f"Hostfunc user data handle {handle} not found.")
    bindings.free_hostfunc_user_data(handle)
    HOSTFUNC_USER_DATA_HANDLES.remove(handle)


def free_all_hostfunc_user_data():
    for handle in HOSTFUNC_USER_DATA_HANDLES:
        bindings.free_hostfunc_user_data(handle)
    HOSTFUNC_USER_DATA_HANDLES.clear()


atexit.register(free_all_hostfunc_user_data)
