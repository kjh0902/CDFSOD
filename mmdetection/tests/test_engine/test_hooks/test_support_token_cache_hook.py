from types import SimpleNamespace

from mmengine.runner import Runner

from mmdet.engine.hooks import SupportTokenCacheHook


def test_support_token_cache_hook_builds_cache_once(monkeypatch):
    support_dataloader = SimpleNamespace(
        dataset=SimpleNamespace(metainfo={'classes': ('a', 'b')}))
    monkeypatch.setattr(
        Runner, 'build_dataloader', lambda dataloader, seed: support_dataloader)

    class Model:
        has_support_token_cache = False

        def __init__(self):
            self.calls = []

        def build_support_token_cache(self, dataloader, class_names):
            self.calls.append((dataloader, class_names))
            self.has_support_token_cache = True

    model = Model()
    runner = SimpleNamespace(model=model, seed=42)
    hook = SupportTokenCacheHook(support_dataloader={'dataset': {}})

    hook.before_test(runner)
    hook.before_test(runner)

    assert model.calls == [(support_dataloader, ('a', 'b'))]
