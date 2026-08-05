from types import SimpleNamespace

from mmdet.engine.hooks import SupportTokenCacheHook


def test_support_token_cache_hook_resolves_shots_from_work_dir():
    hook = SupportTokenCacheHook(support_dataloader={})
    runner = SimpleNamespace(
        work_dir='/tmp/work_dirs/NEU-DET_5shot', _load_from=None)

    assert hook._resolve_support_shots(runner) == 5


def test_support_token_cache_hook_prefers_configured_shots():
    hook = SupportTokenCacheHook(
        support_dataloader={}, support_shots=10)
    runner = SimpleNamespace(
        work_dir='/tmp/work_dirs/NEU-DET_1shot', _load_from=None)

    assert hook._resolve_support_shots(runner) == 10
