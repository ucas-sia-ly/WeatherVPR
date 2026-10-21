import unittest, torch
from weather_vpr.inference.evaluate import recall_at_k
class EvalTests(unittest.TestCase):
    def test_top1(self):
        g=torch.eye(4)
        q=g[[1,3]]
        scores=recall_at_k(g,['a','b','c','d'],q,['b','d'])
        self.assertEqual(scores['R@1'],1.)
    def test_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_k(torch.eye(2),['a','b'],torch.eye(2),['z','b'])
if __name__=='__main__':unittest.main()
