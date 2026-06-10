import tensorrt_llm.bindings as _tb

_bm = _tb.internal.batch_manager

# The per-request KV transfer activity log is populated from the C++ transfer
# paths (CacheSender/CacheReceiver); there is no Python `record` entry point.
# These tests therefore exercise the read/maintenance bindings as a contract /
# smoke test: that the three functions are registered, callable, and return the
# documented types for ids that were never recorded.

# An id that the C++ side will not have recorded in a unit-test process.
_UNKNOWN_REQUEST_ID = 0xDEAD_BEEF_CAFE


def test_dump_unknown_request_id_returns_empty_string():
    dumped = _bm.dump_kv_transfer_activity_log(_UNKNOWN_REQUEST_ID)
    assert isinstance(dumped, str)
    assert dumped == ""


def test_release_unknown_request_id_is_idempotent():
    # Releasing an id that was never recorded must not raise.
    _bm.release_kv_transfer_activity_log(_UNKNOWN_REQUEST_ID)
    _bm.release_kv_transfer_activity_log(_UNKNOWN_REQUEST_ID)


def test_tracked_request_count_returns_non_negative_int():
    count = _bm.kv_transfer_activity_log_tracked_count()
    assert isinstance(count, int)
    assert count >= 0


def test_dump_after_release_is_empty():
    _bm.release_kv_transfer_activity_log(_UNKNOWN_REQUEST_ID)
    assert _bm.dump_kv_transfer_activity_log(_UNKNOWN_REQUEST_ID) == ""
