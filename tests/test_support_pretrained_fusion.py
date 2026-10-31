"""Neck extraction and FE-to-support transfer through actual checkpoint loaders."""
import copy
from unittest.mock import patch
import unittest

import torch
from torch import nn

from visual_test_support import visual, stats_module
from support_transformer_fixture import Encoder, SupportAttention
from test_support_visual import SupportModel


class Model(visual.SupportVisualMixin, nn.Module):
    def __init__(self, support=True, layers=6):
        super().__init__()
        self.support_visual_cfg = {} if support else None
        self.encoder = Encoder(layers=layers, support=support)
        self._support_memory = None


class PretrainedFusionTests(unittest.TestCase):
    def source_state(self):
        torch.manual_seed(71)
        source = Model(False)
        with torch.no_grad():
            for i, layer in enumerate(source.encoder.fusion_layers):
                for p in layer.parameters():
                    p.uniform_(.01 + .01*i, .2 + .01*i)
        return source.state_dict()

    def assert_copied(self, model, state):
        for i, layer in enumerate(model.encoder.support_layers):
            fusion = model.encoder.fusion_layers[i]
            for dst, src in layer.fe_parameter_map().items():
                actual = layer.state_dict()[dst]
                expected = state[f'encoder.fusion_layers.{i}.{src}']
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertNotEqual(actual.data_ptr(), fusion.state_dict()[src].data_ptr())
                self.assertNotEqual(actual.data_ptr(), expected.data_ptr())

    def test_pytorch_and_mmengine_transfer_and_independence(self):
        from mmengine.runner.checkpoint import load_state_dict
        state = self.source_state()
        original = copy.deepcopy(state)
        for loader in (lambda m,s: m.load_state_dict(s, strict=True),
                       lambda m,s: load_state_dict(m, s, strict=True)):
            model = Model()
            with patch('mmengine.logging.print_log') as log:
                loader(model, state)
                self.assertEqual(log.call_count, 6)
                self.assertIn('copied pretrained FE', log.call_args.args[0])
            self.assert_copied(model, state)
            for key in state:
                torch.testing.assert_close(state[key], original[key], rtol=0, atol=0)
            before = copy.deepcopy(model.state_dict())
            with torch.no_grad():
                for p in model.encoder.support_layers[0].parameters():
                    p.add_(1)
            for key,value in model.state_dict().items():
                if not key.startswith('encoder.support_layers.0.'):
                    torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_restoration_preserves_finetuned_support_and_clears_cache(self):
        from mmengine.runner.checkpoint import load_state_dict
        model = Model()
        model.load_state_dict(self.source_state())
        with torch.no_grad():
            for p in model.encoder.support_layers.parameters():
                p.add_(.37)
        state = copy.deepcopy(model.state_dict())
        for loader in (lambda m: m.load_state_dict(state),
                       lambda m: load_state_dict(m,state,strict=True)):
            model._support_memory = {'tokens':torch.ones(1)}
            with patch('mmengine.logging.print_log') as log:
                loader(model)
                self.assertIn('no FE reinitialization', log.call_args.args[0])
            self.assertIsNone(model._support_memory)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,state[key],rtol=0,atol=0)

    def test_nested_model_prefix_and_disabled_loading(self):
        from mmengine.runner.checkpoint import load_state_dict
        state=self.source_state()
        wrapper=nn.Module();wrapper.detector=Model()
        nested={'detector.'+k:v for k,v in state.items()}
        for loader in (lambda:wrapper.load_state_dict(nested),
                       lambda:load_state_dict(wrapper,nested,strict=True)):
            loader()
            self.assert_copied(wrapper.detector,state)
        disabled=Model(False)
        disabled.load_state_dict(state,strict=True)
        self.assertFalse(any('support' in k for k in disabled.state_dict()))

    def test_key_bias_does_not_change_attention_output(self):
        torch.manual_seed(91)
        layer=SupportAttention(8,2)
        q,k=torch.randn(1,5,8),torch.randn(1,13,8)
        before=layer.attn(q,k,k)[0]
        layer.attn.k_proj.bias.data.add_(torch.randn_like(layer.attn.k_proj.bias)*3)
        torch.testing.assert_close(layer.attn(q,k,k)[0],before,rtol=1e-5,atol=1e-6)

    def test_partial_legacy_wrong_shape_and_missing_source_fail_even_nonstrict(self):
        from mmengine.runner.checkpoint import load_state_dict
        state = self.source_state()
        partial = dict(state, **{'encoder.support_layers.0.gamma':torch.ones(8)})
        legacy = dict(state, **{'encoder.support_layers.0.attn.in_proj_weight':torch.ones(24,8)})
        missing = dict(state); del missing['encoder.fusion_layers.3.attn.l_proj.weight']
        complete = Model();complete.load_state_dict(state)
        wrong = copy.deepcopy(complete.state_dict())
        wrong['encoder.support_layers.2.attn.q_proj.weight'] = torch.ones(8,8)
        for bad in (partial, legacy, missing, wrong):
            for loader in (lambda m: m.load_state_dict(bad,strict=False),
                           lambda m: load_state_dict(m,bad,strict=False)):
                with self.assertRaisesRegex(RuntimeError,'support|support layer'):
                    loader(Model())

    def test_copied_text_direction_matches_original_fusion_in_fp32_and_amp(self):
        model=Model();model.load_state_dict(self.source_state())
        torch.manual_seed(14)
        text=torch.randn(2,5,8);support=torch.randn(7,8)
        for amp in (False,True):
            for i in range(6):
                layer=model.encoder.support_layers[i]
                fusion=model.encoder.fusion_layers[i].eval()
                with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
                    normalized_text=fusion.layer_norm_l(text)
                    normalized_support=fusion.layer_norm_v(support)[None].expand(2,-1,-1)
                    update=fusion.attn(normalized_support,normalized_text)[1]
                    expected=text+fusion.gamma_l*update
                    actual=layer(text,support,torch.zeros(2,5,dtype=torch.long),(7,))
                torch.testing.assert_close(actual,expected,
                                           rtol=.015 if amp else 1e-5,
                                           atol=.002 if amp else 1e-6)

    def test_full_size_shapes_gradients_and_identity(self):
        layer=SupportAttention(256,4)
        self.assertEqual(layer.attn.q_proj.weight.shape,(1024,256))
        self.assertEqual(layer.attn.out_proj.weight.shape,(256,1024))
        text=torch.randn(2,4,256,requires_grad=True)
        support=torch.randn(13,256,requires_grad=True)
        classes=torch.tensor([[-1,0,1,-1]]*2)
        for amp in (False,True):
            layer.zero_grad(set_to_none=True)
            with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
                output=layer(text,support,classes,(6,7))
                loss=(output*torch.randn_like(output)).sum()
            loss.backward()
            for name in ('text_norm.weight','support_norm.weight','attn.q_proj.weight',
                         'attn.k_proj.weight','attn.v_proj.weight','attn.out_proj.weight','gamma'):
                grad=dict(layer.named_parameters())[name].grad
                self.assertIsNotNone(grad,name)
                self.assertTrue(torch.isfinite(grad).all(),name)
                self.assertGreater(grad.abs().sum().item(),0,name)
            self.assertIsNone(support.grad)
            torch.testing.assert_close(output[:,[0,3]],text[:,[0,3]],rtol=0,atol=0)
        layer.gamma.data.zero_()
        torch.testing.assert_close(layer(text,support,classes,(6,7)),text,rtol=0,atol=0)

    def test_projected_entropy_and_ungated_ratio_match_forward(self):
        layer=SupportAttention(8,2,gamma_init=.2)
        query=torch.randn(12,8)*7+3;support=torch.randn(51,8)
        nq,ns=layer.text_norm(query),layer.support_norm(support)
        update,prob=layer.attn(nq[None],ns[None],ns[None],need_weights=True,
                               average_attn_weights=False)
        expected=-(prob[0,:,:8]*prob[0,:,:8].clamp_min(1e-30).log()).sum(-1)/torch.tensor(51.).log()
        entropy=stats_module.VisualTokenStats.attention_entropy(layer.attn,nq[:8],ns,7)
        torch.testing.assert_close(entropy,expected,rtol=1e-5,atol=1e-6)
        stats=stats_module.VisualTokenStats(1,'cpu',1,2)
        stats.record_gamma(0,layer.gamma)
        stats.cross_attention(0,layer.attn,nq,ns,update[0]*layer.gamma,
                              torch.zeros(12,dtype=torch.long),reference=query,update=update[0])
        row=stats.report(('a',))['support_cross_attention']['0']
        self.assertAlmostEqual(row['residual_contribution']['value'],
                               float((update[0]*layer.gamma).norm()/query.norm()),places=7)
        self.assertAlmostEqual(row['ungated_update_contribution']['value'],
                               float(update.norm()/query.norm()),places=7)
        for i,head in enumerate(row['normalized_attention_entropy_per_head']):
            self.assertEqual(head['count'],8)
            self.assertAlmostEqual(head['value'],float(expected[i].mean()),places=6)

    def test_neck_feature_zero_current_weights_and_mode_restore_on_error(self):
        model=SupportModel().train()
        model.backbone.dropout.eval();model.neck.dropout.eval()
        modules=list(model.backbone.modules())+list(model.neck.modules())
        modes=[m.training for m in modules]
        rng=torch.get_rng_state().clone()
        observed=[]
        handle=model.neck.register_forward_pre_hook(lambda m,a: observed.append(
            (torch.is_grad_enabled(),[x.training for x in m.modules()])))
        first=model.support_memory()['tokens']
        model.neck.weight.data.fill_(3)
        second=model.support_memory()['tokens']
        torch.testing.assert_close(second,first*3)
        self.assertTrue(all(not grad and not any(states) for grad,states in observed))
        self.assertIsNone(model.neck.weight.grad)
        self.assertEqual(modes,[m.training for m in modules])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with patch.object(model.neck,'forward',side_effect=RuntimeError('neck failed')):
            with self.assertRaisesRegex(RuntimeError,'neck failed'):
                model.refresh_support_memory()
        self.assertIsNone(model._support_memory)
        self.assertEqual(modes,[m.training for m in modules])
        handle.remove()
        # Feature 1 is deliberately different; extraction must use feature 0.
        with patch.object(model.neck,'forward',side_effect=lambda xs:[xs[0]+10,xs[0]+100]):
            memory=model.support_memory()
            torch.testing.assert_close(memory['tokens'],first+10)


if __name__=='__main__':
    unittest.main()
