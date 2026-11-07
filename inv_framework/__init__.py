"""Inverse-problem library; domain backends load only when requested."""
__version__ = "0.1.0"
__all__ = ["CTTaskRequest", "CTTaskResult", "list_capabilities", "run_ct_task"]


def __getattr__(name):
    if name in __all__:
        from .ct import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
