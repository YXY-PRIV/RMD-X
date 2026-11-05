#!/usr/bin/env python3
"""Small counterexample contracts, not AP or detector reproduction."""
from pathlib import Path
import sys,unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from portable.core import project,choose,mix_visual,gaussian

class PairContract(unittest.TestCase):
 def test_supported_conflict_and_exact_fallback(self):
  n=8;valid=np.ones((n,2),bool);kept=np.ones((n,2),bool)
  mapping=np.tile(np.array([0,1],np.int64),(n,1));cats=np.tile(np.array([0,2],np.int64),(n,1));ci=np.tile(np.array([0,1],np.int64),(n,1));branch=cats.copy()
  availability=np.ones((n,2),np.float32);T=np.full(n,3,np.int64)
  delta=np.tile(np.array([.3,-.1],np.float32),(n,1))
  mapping[1]=[1,0];cats[1]=[2,0];ci[1]=[1,0];delta[1]=[.1,.4]
  valid[2]=[True,False];mapping[2]=[0,0];cats[2]=[0,-1];ci[2]=[0,-1];branch[2]=[0,0];availability[2]=[1,0];delta[2]=[.2,0]
  valid[3]=[True,False];kept[3]=[False,True];mapping[3]=[-1,0];cats[3]=[2,-1];ci[3]=[1,-1];branch[3]=[-1,2];availability[3]=[1,0];delta[3]=[.2,0]
  availability[4]=[1,0];delta[4]=[.3,0]
  T[5]=1;availability[5]=[1,0];delta[5]=[.3,0]
  T[6]=0;availability[6]=[0,0];delta[6]=[0,0]
  delta[7]=[.1,.1]
  out,active,D,stats=project(delta,valid,availability,T,ci,cats,mapping,kept,branch)
  np.testing.assert_array_equal(active,[True,True,False,False,False,False,False,True])
  np.testing.assert_array_equal(out[~active].view(np.uint32),0)
  np.testing.assert_array_equal(out[active].astype(np.float64).sum(1),0)
  # A negative D increases V and decreases F: F is not always boosted.
  self.assertLess(float(D[0]),0);self.assertGreater(float(out[0,0]),0);self.assertLess(float(out[0,1]),0)
  for row in np.flatnonzero(active):
   v,f=mapping[row]
   self.assertLessEqual(abs(float(out[row,f])-float(out[row,v])-(float(delta[row,f])-float(delta[row,v]))),2**-22)
  self.assertGreater(stats['active_CID0_rows'],0);self.assertEqual(stats['active_zero_D_rows'],1)
 def test_fixed_anchor_ties_and_missing_branch(self):
  kept=np.array([[1,1],[1,1],[0,1],[1,0],[0,0]],bool)
  q=np.array([[.5,.5],[.2,.3],[0,.1],[.2,0],[0,0]],np.float32)
  np.testing.assert_array_equal(choose(kept,q),[0,1,1,0,-1])
 def test_mix_masks_and_clip_boundary(self):
  M=np.array([[1,0],[1,1]],bool);base=np.array([[0,.8],[1,.25]],np.float32);head=np.array([[0,.3],[1,1]],np.float32)
  out=mix_visual(base,head,M);self.assertEqual(out.dtype,np.float32)
  self.assertEqual(float(out[0,1]),0);self.assertAlmostEqual(float(out[0,0]),1e-6,places=12);self.assertEqual(float(out[1,0]),1);self.assertEqual(float(out[1,1]),.5)
 def test_gaussian_tie_order_and_group_isolation(self):
  scores=np.array([.5,.5,.5]);boxes=np.array([[0,0,10,10],[0,0,10,10],[0,0,10,10]])
  images=np.array([1,1,1]);cats=np.array([1,1,2]);rows=np.array([4,2,5])
  before=boxes.copy();out,rank=gaussian(scores,boxes,images,cats,rows)
  self.assertEqual(rank[1],0);self.assertEqual(out[1],.5);self.assertAlmostEqual(out[0],.5*np.exp(-1));self.assertEqual(out[2],.5)
  np.testing.assert_array_equal(scores,[.5,.5,.5]);np.testing.assert_array_equal(boxes,before)
if __name__=='__main__':unittest.main()
