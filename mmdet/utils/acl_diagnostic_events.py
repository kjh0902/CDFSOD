"""Read-only diagnostic events; inactive outside a validation batch."""
from contextvars import ContextVar


_sink = ContextVar('acl_diagnostic_sink', default=None)


def diagnostic_event(event, **payload):
    sink = _sink.get()
    if sink is not None:
        sink(event, **payload)


def activate_diagnostics(sink):
    return _sink.set(sink)


def deactivate_diagnostics(token):
    _sink.reset(token)
