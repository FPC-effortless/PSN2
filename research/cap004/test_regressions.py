from pathlib import Path
import sys,copy,types,torch,numpy as np,pytest
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(REPO/'dpsl_python')]
torch.set_num_threads(1)


def test_psn2_per_example_readout_and_masked_loss_gradient():
 from psn2.core import PSN2System
 torch.manual_seed(101);m=PSN2System(dim=16,max_nodes=8,grid_vocab=11,rel_vocab=5,stage='D1')
 grid=torch.randint(0,10,(4,4,4));target=grid.clone();grid[:,1:3,1:3]=10;mask=torch.zeros_like(grid,dtype=torch.float);mask[:,1:3,1:3]=1
 out=m.forward_batch({'type':'arc','input_grid':grid,'target_grid':target,'loss_mask':mask})
 assert out['logits'].requires_grad;out['loss'].backward();assert m.node_bank.nu.grad is not None
 shape=m.encode_grid(grid);a=m._node_repr(shape);b=torch.cat([m._node_repr(s[None]) for s in shape])
 assert torch.allclose(a,b,atol=1e-6)
