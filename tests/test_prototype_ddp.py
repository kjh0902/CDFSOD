"""Exercise the actual encoder loop and fusion under CPU DDP."""
import ast
from datetime import timedelta
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import test_qwen_offline_sanity as sanity


def fusion_layer():
    path = sanity.ROOT / 'mmdet/models/utils/vlfuse_helper.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'BiMultiHeadAttention', 'BiAttentionBlock', 'SingleScaleBiAttentionBlock'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    env = dict(torch=torch, nn=nn, F=torch.nn.functional, MAX_CLAMP_VALUE=50000)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env['SingleScaleBiAttentionBlock'](3, 3, 6, 1, dropout=0.)


class CheckpointFusion(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, visual_feature, lang_feature, **kwargs):
        return checkpoint(lambda v, t: self.layer(v, t, **kwargs),
                          visual_feature, lang_feature, use_reentrant=True)


class PrototypeLoss(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, features):
        text = self.detector.build_prototype_text_dict(2, 'cpu')
        output = self.detector.forward_encoder(
            feat=features, feat_mask=torch.zeros(2, 4, dtype=torch.bool),
            feat_pos=torch.zeros_like(features), spatial_shapes=torch.tensor([[2, 2]]),
            level_start_index=torch.tensor([0]), valid_ratios=torch.ones(2, 1, 2),
            text_dict=text)
        scores = output['memory'] @ output['memory_text'].transpose(1, 2)
        return scores.square().mean() + 0.1 * sanity.etf.nearest_etf_loss(
            output['memory_text'])


@unittest.skipUnless(torch.distributed.is_available() and
                     torch.distributed.is_gloo_available(), 'Requires CPU Gloo')
class PrototypeDDPTests(unittest.TestCase):
    def setUp(self):
        self.fixture = sanity.QwenSanity()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_all_enhancer_parameters_receive_grad_for_two_iterations(self):
        for use_checkpoint in (False, True):
            with self.subTest(checkpoint=use_checkpoint):
                torch.manual_seed(7)
                model = self.fixture.model(
                    entries={'pitted_surface': 'small uneven holes',
                             'other': 'dark round shape', 'beetles': 'shiny wings'},
                    names=['pitted_surface', 'other', 'beetles'])
                model.encoder = sanity.recording_encoder()
                model.encoder.fusion_layers = nn.ModuleList([fusion_layer() for _ in range(6)])
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                if use_checkpoint:
                    model.encoder.fusion_layers = nn.ModuleList([
                        CheckpointFusion(layer) for layer in model.encoder.fusion_layers])
                loss_model = PrototypeLoss(model)
                features = torch.randn(2, 4, 3)
                store = torch.distributed.TCPStore(
                    '127.0.0.1', 0, 1, True, use_libuv=False)
                torch.distributed.init_process_group(
                    'gloo', store=store, rank=0, world_size=1,
                    timeout=timedelta(seconds=30))
                try:
                    ddp = nn.parallel.DistributedDataParallel(
                        loss_model, find_unused_parameters=False)
                    optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-4)
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        loss = ddp(features)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        unused = [name for name, p in model.named_parameters()
                                  if p.grad is None]
                        self.assertEqual(unused, [])
                        self.assertGreater(model.encoder.text_layers[-1].attn.in_proj_weight
                                           .grad.abs().sum().item(), 0)
                        optimizer.step()
                finally:
                    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
