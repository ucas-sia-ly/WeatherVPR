"""CPU-only tests: no DINOv2 download/checkpoint required."""
import unittest
import torch
from torch import nn
from weather_vpr.models.adapter import PatchCLSAdapter, WeatherAdapterSALAD
from weather_vpr.losses.distillation import (weighted_patch_losses, descriptor_alignment,
                                            cross_domain_infonce)

class FakeBackbone(nn.Module):
    def forward(self, x):
        p = x.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1)
        return p, x.mean(dim=(2,3)).mean(1, keepdim=True).repeat(1,8)

class FakeSalad(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.config = {'agg_config': {'num_channels':8}}
        self.proj = nn.Linear(8, 8)
    def aggregator(self, x, return_aux=True):
        p, cls = x
        keep = torch.sigmoid(p.mean(1).flatten(1))
        d = self.proj(p.mean(dim=(2,3)) + cls)
        if return_aux:
            return d, {'backbone_local_features':p, 'assignment_keep':keep}
        return d
    def forward(self, x, return_aux=True):
        return self.aggregator(self.backbone(x), return_aux)

class Tests(unittest.TestCase):
    def test_initial_identity_and_gradient(self):
        torch.manual_seed(1)
        base = FakeSalad()
        net = WeatherAdapterSALAD(base, ['night','rain'], bottleneck=4, adapter_type='output')
        im = torch.randn(2,3,4,4)
        t, ta = net.teacher(im)
        s, sa = net.student(im, 'night')
        self.assertTrue(torch.allclose(t, s, atol=1e-6))
        idx = torch.arange(16).expand(2,-1)
        q = torch.ones(2,16)
        losses = weighted_patch_losses(sa, ta, idx, q)
        loss = descriptor_alignment(s, t+0.2) + losses['patch'] + .1 * losses['keep']
        loss.backward()
        self.assertFalse(any(x.grad is not None for x in net.base.parameters()))
        self.assertTrue(any(x.grad is not None for x in net.adapters['night'].parameters()))
        self.assertFalse(any(x.grad is not None for x in net.adapters['rain'].parameters()))
    def test_invalid_correspondence_rejected(self):
        net = WeatherAdapterSALAD(FakeSalad(), ['night'],4, adapter_type='output')
        x = torch.randn(1,3,4,4)
        _, ta = net.teacher(x)
        _, sa = net.student(x,'night')
        idx = torch.full((1,16),-1,dtype=torch.long)
        with self.assertRaises(ValueError):
            weighted_patch_losses(sa,ta,idx,torch.ones(1,16))
        out = weighted_patch_losses(sa,ta,idx,torch.zeros(1,16))
        self.assertEqual(out['positive_patches'].item(),0)
        self.assertEqual(out['patch'].item(),0.)
    def test_contrastive(self):
        a = torch.eye(3)
        same = cross_domain_infonce(a,a,[('A','1'),('B','2'),('C','3')])
        flipped = cross_domain_infonce(a,a.flip(0),[('A','1'),('B','2'),('C','3')])
        self.assertLess(same.item(),flipped.item())

if __name__=='__main__':
    unittest.main()
