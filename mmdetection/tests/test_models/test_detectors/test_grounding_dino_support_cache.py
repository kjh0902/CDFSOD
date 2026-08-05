from types import SimpleNamespace

import torch
import torch.nn as nn

from mmdet.models.detectors.grounding_dino import GroundingDINO


class SupportCacheHarness(nn.Module):
    _generate_support_tokens = GroundingDINO._generate_support_tokens
    generate_training_support_tokens = \
        GroundingDINO.generate_training_support_tokens
    build_support_token_cache = GroundingDINO.build_support_token_cache

    def __init__(self):
        super().__init__()
        self.textualized_visual_token_generator = nn.Identity()
        self.use_autocast = False
        self._support_dataloader = None
        self._training_support_class_names = None
        self._support_visual_tokens = None
        self._support_visual_token_labels = None
        self._support_class_names = None

    def data_preprocessor(self, data_batch, training=False):
        return data_batch

    def extract_feat(self, batch_inputs):
        return batch_inputs

    def generate_textualized_visual_tokens(self, visual_features,
                                            batch_data_samples):
        return visual_features


def make_support_batch(labels, token_values):
    labels = torch.tensor(labels, dtype=torch.long)
    tokens = torch.tensor(token_values, dtype=torch.float32).unsqueeze(-1)
    tokens = tokens.repeat(1, 768)
    data_sample = SimpleNamespace(
        gt_instances=SimpleNamespace(labels=labels))
    return dict(inputs=tokens, data_samples=[data_sample])


def test_support_cache_contains_every_object_in_class_major_order():
    model = SupportCacheHarness()
    support_dataloader = [
        make_support_batch([1, 0, 0], [10, 20, 30]),
        make_support_batch([0], [40]),
        make_support_batch([1], [50]),
    ]

    model.build_support_token_cache(
        support_dataloader, class_names=('a', 'b'))

    assert model._support_visual_tokens[:, 0].tolist() == [20, 30, 40, 10, 50]
    assert model._support_visual_token_labels.tolist() == [0, 0, 0, 1, 1]
    assert model._support_class_names == ('a', 'b')


def test_training_support_tokens_are_regenerated_every_call():

    class CountingSupportLoader:

        def __init__(self, batches):
            self.batches = batches
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.batches)

    model = SupportCacheHarness()
    support_dataloader = CountingSupportLoader([
        make_support_batch([0], [10]),
        make_support_batch([1], [20]),
    ])
    model._support_dataloader = support_dataloader
    model._training_support_class_names = ('a', 'b')

    first_tokens, first_labels, _ = \
        model.generate_training_support_tokens()
    second_tokens, second_labels, _ = \
        model.generate_training_support_tokens()

    assert support_dataloader.iterations == 2
    assert first_tokens[:, 0].tolist() == [10, 20]
    assert second_tokens[:, 0].tolist() == [10, 20]
    assert first_labels.tolist() == [0, 1]
    assert second_labels.tolist() == [0, 1]
