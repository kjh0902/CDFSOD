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
            self.train_calls = []
            self.test_calls = []

        def set_support_dataloader(self, dataloader, class_names):
            self.train_calls.append((dataloader, class_names))

        def build_support_token_cache(self, dataloader, class_names):
            self.test_calls.append((dataloader, class_names))
            self.has_support_token_cache = True

    model = Model()
    runner = SimpleNamespace(model=model, seed=42)
    hook = SupportTokenCacheHook(support_dataloader={'dataset': {}})

    hook.before_train(runner)
    hook.before_test(runner)
    hook.before_test(runner)

    assert model.train_calls == [(support_dataloader, ('a', 'b'))]
    assert model.test_calls == [(support_dataloader, ('a', 'b'))]
