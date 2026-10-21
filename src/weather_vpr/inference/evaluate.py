"""Fixed day gallery, real-weather query Recall@K; uses one shared base network."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from weather_vpr.utils.backend import project_root,enable_backend_imports
from weather_vpr.models.adapter import WeatherAdapterSALAD
from weather_vpr.data.evaluation_dataset import LabeledImages

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for blk in iter(lambda:f.read(1024*1024),b''): h.update(blk)
    return h.hexdigest()

@torch.inference_mode()
def extract(loader,net,weather,device):
    vectors=[]; labels=[]
    for x,y in loader:
        x=x.to(device)
        desc=(net.teacher(x,return_aux=False) if weather=='day' else
              net.student(x,weather,return_aux=False))
        vectors.append(F.normalize(desc.float(),dim=1).cpu())
        labels.extend(y)
    return torch.cat(vectors,dim=0),labels

def recall_at_k(gallery_vectors, gallery_ids, query_vectors, query_ids, ks=(1,5,10), chunk=128):
    if len(gallery_ids)!=len(gallery_vectors) or len(query_ids)!=len(query_vectors):
        raise ValueError('Vector and label count mismatch')
    gallery_set=set(gallery_ids)
    unknown=sorted(set(query_ids)-gallery_set)
    if unknown:
        raise ValueError(f'Query place_id not present in gallery (first five): {unknown[:5]}')
    k=max(ks); k=min(k,len(gallery_ids)); correct={n:0 for n in ks}
    ref=F.normalize(gallery_vectors.float(),dim=1)
    qry=F.normalize(query_vectors.float(),dim=1)
    for start in range(0,len(qry),chunk):
        score=qry[start:start+chunk]@ref.T
        chosen=score.topk(k,dim=1).indices.tolist()
        for index,pred in enumerate(chosen):
            true_id=query_ids[start+index]
            for n in ks:
                if any(gallery_ids[t]==true_id for t in pred[:n]): correct[n]+=1
    return {f'R@{n}': correct[n]/len(qry) for n in ks}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=project_root()/'weights/dino_salad.ckpt')
    p.add_argument('--backbone-repo',type=Path,default=project_root()/'third_party/dinov2')
    p.add_argument('--gallery-csv',type=Path,default=project_root()/'data/test/gallery.csv')
    p.add_argument('--query-csv',type=Path,required=True)
    p.add_argument('--weather',default='night',help='day or trained adapter weather')
    p.add_argument('--adapter-dir',type=Path,help='Training output dir for this weather')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path,default=project_root()/'runs/evaluation_night.json')
    a=p.parse_args()
    if a.device=='cpu': os.environ['XFORMERS_DISABLED']='1'
    if a.weather!='day' and a.adapter_dir is None:
        p.error('--adapter-dir is needed when --weather is not day')
    enable_backend_imports()
    from workflow.model import load_checkpoint_model, input_preprocessing
    base=load_checkpoint_model(a.checkpoint,device=a.device,backbone_repo=a.backbone_repo)
    cfg=None
    if a.adapter_dir:
        cfg=json.loads((a.adapter_dir/'adapter.json').read_text(encoding='utf8'))
        if cfg['weather']!=a.weather or cfg['checkpoint_sha256']!=sha256(a.checkpoint) or \
           cfg['model_config']!=base.config or cfg['preprocessing']!=input_preprocessing(base.image_size):
            raise ValueError('Adapter does not match base model, weather, or preprocessing')
    net=WeatherAdapterSALAD(base,(a.weather,) if cfg else (), bottleneck=cfg['bottleneck'] if cfg else 64).to(a.device).eval()
    if cfg:
        state=load_file(str(a.adapter_dir/'adapter.safetensors'),device='cpu')
        net.adapters[a.weather].load_state_dict(state,strict=True)
    gallery_ds=LabeledImages(a.gallery_csv,base.image_size)
    query_ds=LabeledImages(a.query_csv,base.image_size)
    g,gl=extract(DataLoader(gallery_ds,batch_size=a.batch_size),net,'day',a.device)
    q,ql=extract(DataLoader(query_ds,batch_size=a.batch_size),net,a.weather,a.device)
    recalls=recall_at_k(g,gl,q,ql)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result={'gallery_size':len(gl),'query_size':len(ql),'weather':a.weather,
            'metrics':recalls,'gallery_csv':str(a.gallery_csv.resolve()),
            'query_csv':str(a.query_csv.resolve()),'adapter_dir':str(a.adapter_dir) if a.adapter_dir else None,
            'warning':'Labels are user-supplied place IDs; test leakage and geographic ground truth must be independently checked'}
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
