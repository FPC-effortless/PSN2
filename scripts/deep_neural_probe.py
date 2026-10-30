"""Actual-source CPU diagnostics on constructed inputs; no optimizer or training."""
import copy,hashlib,io,json,random,sys,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from psn2.core import PSN2System
torch.set_num_threads(1)
def seed():random.seed(101);np.random.seed(101);torch.manual_seed(101)
def create():
 seed();return PSN2System(dim=32,max_nodes=16,grid_vocab=10,rel_vocab=5,stage='D1').eval()
def batch():return {'type':'arc','input_grid':torch.tensor([[[1,0],[0,0]],[[0,1],[0,0]]]),'target_grid':torch.tensor([[[0,1],[0,0]],[[1,0],[0,0]]])}
rows=[]
def check(name,fn):
 try:rows.append({'check':name,**fn()})
 except Exception as e:rows.append({'check':name,'status':'ERROR','error':str(e),'traceback':traceback.format_exc()})
def spatial():
 m=create()
 with torch.no_grad():out=m.forward_batch(batch())
 pred=out['pred'];difference=float((pred[0,0,1]-pred[0,1,0]).abs().max())
 return {'status':'FAIL' if difference<1e-7 else 'MEASURED','same_color_logit_max_difference':difference,'finite':bool(torch.isfinite(pred).all()),'scope':'Same-color cells need different targets in a reflection; this is a constructed structural diagnostic.'}
def gradients():
 m=create();captured=[]
 handle=m.grid_decoder.register_forward_hook(lambda module,args,out:captured.append(out))
 try:m.forward_batch(batch())
 finally:handle.remove()
 logits=captured[-1];loss=F.cross_entropy(logits.reshape(-1,10),batch()['target_grid'].reshape(-1))
 names,params=zip(*m.named_parameters());grads=torch.autograd.grad(loss,params,allow_unused=True)
 missing=[n for n,g in zip(names,grads) if g is None]
 zero=[n for n,g in zip(names,grads) if g is not None and torch.count_nonzero(g)==0]
 nonfinite=[n for n,g in zip(names,grads) if g is not None and not torch.isfinite(g).all()]
 return {'status':'FAIL' if nonfinite else 'MEASURED','primary_loss':float(loss.detach()),'trainable_parameters':sum(p.numel() for p in params if p.requires_grad),'total_parameters':sum(p.numel() for p in params),'parameter_tensors':len(params),'primary_connected_tensors':sum(g is not None for g in grads),'primary_disconnected':missing,'zero':zero,'nonfinite':nonfinite,'scope':'Grid primary loss only; unused graph/optional heads need module-specific interpretation.'}
def replay():
 m=create()
 with torch.no_grad():m.forward_batch(batch())
 data=io.BytesIO();torch.save(copy.deepcopy(m.state_dict_full()),data);data.seek(0)
 restored=create();restored.load_state_dict_full(torch.load(data,map_location='cpu',weights_only=False))
 seed()
 with torch.no_grad():a=m.forward_batch(batch())['pred']
 seed()
 with torch.no_grad():b=restored.forward_batch(batch())['pred']
 error=float((a-b).abs().max())
 return {'status':'PASS' if error<=1e-6 else 'FAIL','next_forward_max_abs_difference':error,'scope':'Constructed CPU model-state replay, not optimizer/sampler/PRNG training continuation.'}
check('spatial_identifiability',spatial);check('primary_grid_gradients',gradients);check('model_state_replay',replay)
report={'study':'DEEP-FIX001','qualification':'CONSTRUCTED_ACTUAL_SOURCE_NO_FIT','torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'source_sha256':hashlib.sha256((ROOT/'psn2/core.py').read_bytes()).hexdigest(),'checks':rows,'optimizer_steps':0,'gpu_minutes':0,'local_plasticity_may_execute':True,'architecture_superiority_eligible':False}
p=Path(sys.argv[1] if len(sys.argv)>1 else 'deep-neural-results.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if any(r['status']=='ERROR' for r in rows):raise SystemExit(1)
