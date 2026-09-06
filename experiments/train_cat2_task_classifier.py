#!/usr/bin/env python
"""8-class surgical TASK classifier for the Cat-2 grounding spine.

WHY A TASK CLASSIFIER AND NOT AN INSTRUMENT DETECTOR (measured 2026-08-20, zero GPU):
conditioning the fine tool distribution on the task raises expected per-question accuracy over the
top-10 tools by +0.1725 with an oracle task label -- Large Needle Driver 45% marginal -> 97% in
"retraction and collision avoidance" vs 3-6% elsewhere; Cadiere Forceps 57% -> 97% in "uterine
horn" vs 0% in retraction. Meanwhile the tools that do NOT move with task (Mega Needle Driver,
30 deg Endoscope) are precisely the near-visual-duplicates a detector could never separate. So the
learnable signal is the TASK, and the tool list follows from P(tool | task).

SPLIT IS BY CASE, NOT BY CLIP. Clips from one case share the same installed instruments and the
same surgical context; a clip-level split would leak and report a fantasy accuracy.
"""
import argparse, csv, glob, json, os, random, sys, time
import numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision as tv

LAB="/scratch/sc20osc/miccai-2026/SurgVU/data/external/cat2_train_labels/SURGVU25_train_labels"
VID="/scratch/sc20osc/miccai-2026/SurgVU/data/surgvu24_videos/surgvu24"
TASKS=["other","range of motion","rectal artery/vein","retraction and collision avoidance",
       "skills application","suspensory ligaments","suturing","uterine horn"]
T2I={t:i for i,t in enumerate(TASKS)}

def load_segments():
    parts={}
    for d in sorted(glob.glob(os.path.join(VID,"case_*"))):
        case=os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d,"*_video_part_*.mp4"))):
            try: parts[(case,int(os.path.splitext(p)[0].rsplit("_",1)[-1]))]=p
            except ValueError: pass
    segs=[]
    for d in sorted(glob.glob(os.path.join(LAB,"case_*"))):
        case=os.path.basename(d); f=os.path.join(d,"tasks.csv")
        if not os.path.exists(f): continue
        seen=set()
        for r in csv.DictReader(open(f)):
            sig=(r.get("index"),r.get("start_part"),r.get("start_time"),r.get("groundtruth_taskname"))
            if sig in seen: continue
            seen.add(sig)
            tk=" ".join((r.get("groundtruth_taskname") or "").lower().split())
            if tk not in T2I: continue
            try:
                sp=int(float(r["start_part"])); a=float(r["start_time"]); b=float(r["stop_time"])
            except Exception: continue
            v=parts.get((case,sp))
            if v and b>a: segs.append((case,v,a,b,T2I[tk]))
    return segs

class Clips(Dataset):
    def __init__(self, segs, k=4, train=True, size=224):
        self.segs=segs; self.k=k; self.train=train
        self.tf = tv.transforms.Compose(
            ([tv.transforms.RandomResizedCrop(size,scale=(0.7,1.0)),
              tv.transforms.RandomHorizontalFlip()] if train else
             [tv.transforms.Resize(int(size*1.14)), tv.transforms.CenterCrop(size)]) +
            [tv.transforms.ToTensor(),
             tv.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.segs)
    def __getitem__(self, i):
        case,v,a,b,y=self.segs[i]
        cap=cv2.VideoCapture(v); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        ts=(np.random.uniform(a,b,self.k) if self.train else np.linspace(a,b,self.k+2)[1:-1])
        ims=[]
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps))
            ok,fr=cap.read()
            if not ok: fr=np.zeros((224,224,3),np.uint8)
            ims.append(self.tf(tv.transforms.functional.to_pil_image(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))))
        cap.release()
        return torch.stack(ims), y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--k",type=int,default=4)
    ap.add_argument("--bs",type=int,default=8); ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--out",default="/scratch/sc20osc/miccai-2026/SurgVU/models/cat2_task_cls")
    a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    segs=load_segments(); cases=sorted({s[0] for s in segs})
    random.Random(0).shuffle(cases)
    nval=max(15,len(cases)//5); val_cases=set(cases[:nval])          # CASE-disjoint
    tr=[s for s in segs if s[0] not in val_cases]; va=[s for s in segs if s[0] in val_cases]
    print("segments %d | cases %d | train %d (%d cases) | val %d (%d cases)"%(
          len(segs),len(cases),len(tr),len(cases)-nval,len(va),nval),flush=True)
    cnt=np.bincount([s[4] for s in tr],minlength=len(TASKS))
    print("train class counts:", dict(zip(TASKS,cnt.tolist())),flush=True)

    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc=nn.Linear(m.fc.in_features,len(TASKS)); m=m.to(dev)
    w=torch.tensor((cnt.sum()/np.maximum(cnt,1))**0.5,dtype=torch.float32,device=dev)  # mild rebalance
    crit=nn.CrossEntropyLoss(weight=w/w.mean())
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4)
    dl=DataLoader(Clips(tr,a.k,True),batch_size=a.bs,shuffle=True,num_workers=6,drop_last=True)
    dv=DataLoader(Clips(va,a.k,False),batch_size=a.bs,shuffle=False,num_workers=6)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr*5,total_steps=a.epochs*max(1,len(dl)))
    best=0.0
    for ep in range(a.epochs):
        m.train(); t0=time.time()
        for x,y in dl:
            B,K=x.shape[:2]
            x=x.view(B*K,*x.shape[2:]).to(dev,non_blocking=True); y=y.to(dev)
            loss=crit(m(x).view(B,K,-1).mean(1),y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.eval(); ok=0; n=0; per=np.zeros((len(TASKS),2))
        with torch.no_grad():
            for x,y in dv:
                B,K=x.shape[:2]
                p=m(x.view(B*K,*x.shape[2:]).to(dev)).view(B,K,-1).mean(1).argmax(1).cpu()
                ok+=(p==y).sum().item(); n+=len(y)
                for t,q in zip(y.tolist(),p.tolist()):
                    per[t,1]+=1; per[t,0]+= (t==q)
        acc=ok/max(n,1)
        print("epoch %d  val_acc %.4f  (%.0fs)"%(ep,acc,time.time()-t0),flush=True)
        for i,t in enumerate(TASKS):
            if per[i,1]: print("      %-36s %.3f  (n=%d)"%(t,per[i,0]/per[i,1],int(per[i,1])),flush=True)
        if acc>best:
            best=acc; torch.save({"state":m.state_dict(),"tasks":TASKS,"val_acc":acc},
                                 os.path.join(a.out,"best.pt"))
            print("      saved best (%.4f)"%acc,flush=True)
    json.dump({"best_val_acc":best,"tasks":TASKS,"val_cases":sorted(val_cases),
               "n_train":len(tr),"n_val":len(va),"split":"CASE-disjoint"},
              open(os.path.join(a.out,"result.json"),"w"),indent=1)
    print("BEST VAL ACC %.4f"%best)

if __name__=="__main__": main()
