import glob,os,collections,hashlib,shutil,sys,cv2
sys.path.insert(0,'.'); from lib import read_labels,lbl_path
OUT='/home/claude/ds/merged'
def split_of(key):
    h=int(hashlib.md5(key.encode()).hexdigest(),16)%100
    return 'test' if h<10 else 'val' if h<20 else 'train'
def clean(b):
    b=[min(1,max(0,v)) for v in b]
    return b if (b[2]-b[0])>0.004 and (b[3]-b[1])>0.004 else None
stats=collections.Counter(); bad=collections.Counter()
def put(src,split,name,labels):
    os.makedirs(f'{OUT}/images/{split}',exist_ok=True); os.makedirs(f'{OUT}/labels/{split}',exist_ok=True)
    im=cv2.imread(src)
    if im is None: bad['unreadable']+=1; return
    cv2.imwrite(f'{OUT}/images/{split}/{name}.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,92])
    with open(f'{OUT}/labels/{split}/{name}.txt','w') as fh:
        for c,b in labels:
            fh.write(f"{c} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}\n"); stats[(split,c)]+=1
    stats[(split,'img')]+=1
# pallet detection v5 + v6 -> pallet_base (6); group by source photo
groups=collections.defaultdict(list)
for v in ('v5i','v6i'):
    for f in glob.glob(f'raw/pallet detection.{v}.yolov8/*/images/*.jpg'):
        groups[os.path.basename(f).split('.rf.')[0]].append((v,f))
for stem,items in groups.items():
    sp=split_of('pd/'+stem)
    # train: up to one copy from each version; val/test: one copy only
    chosen=[]; seen=set()
    for v,f in sorted(items):
        if v in seen: continue
        seen.add(v); chosen.append((v,f))
        if sp!='train': break
    for i,(v,f) in enumerate(chosen):
        L=read_labels(lbl_path(f)) or []
        labs=[]
        for c,b,_ in L:
            if c!=1: bad['pd_dash_class']+=1; continue
            b=clean(b)
            if b is None: bad['tiny_or_degenerate']+=1; continue
            labs.append((6,b))
        put(f,sp,f'pd_{v}_{hashlib.md5(stem.encode()).hexdigest()[:10]}',labs)
# Warehouse Object Detection: Product-Pallet (4) -> pallet (0)
for f in glob.glob('raw/Warehouse Object Detection.v1i.yolov8/*/images/*.jpg'):
    stem=os.path.basename(f).split('.rf.')[0]
    labs=[(0,clean(b)) for c,b,_ in read_labels(lbl_path(f)) or [] if c==4 and clean(b)]
    put(f,split_of('wod/'+stem),'wod_'+hashlib.md5(stem.encode()).hexdigest()[:10],labs)
print(sorted(stats.items(),key=str)); print(dict(bad))
