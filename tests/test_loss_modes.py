"""Check identity-coordinate toy support and q-mask safety for both modes."""
import unittest
import torch
from weather_vpr.losses.distillation import weighted_patch_losses

class LossModeTests(unittest.TestCase):
    def test_saliency_vs_uniform(self):
        t = torch.zeros(1,2,1,2); t[:,0]=1
        s = t.clone(); s[0,0,0,1]=0; s[0,1,0,1]=1
        day={'backbone_local_features':t,'assignment_keep':torch.tensor([[1.,0.]])}
        student={'backbone_local_features':s,'assignment_keep':torch.tensor([[1.,0.]])}
        idx=torch.tensor([[0,1]])
        q=torch.ones(1,2)
        sal=weighted_patch_losses(student,day,idx,q,importance_mode='saliency')
        uni=weighted_patch_losses(student,day,idx,q,importance_mode='uniform')
        self.assertAlmostEqual(sal['patch'].item(),0.,places=5)
        self.assertGreater(uni['patch'].item(),0.4)
    def test_invalid_mode(self):
        x={'backbone_local_features':torch.ones(1,2,1,1),'assignment_keep':torch.ones(1,1)}
        with self.assertRaises(ValueError):
            weighted_patch_losses(x,x,torch.zeros(1,1,dtype=torch.long),torch.ones(1,1),importance_mode='bad')
if __name__=='__main__': unittest.main()
