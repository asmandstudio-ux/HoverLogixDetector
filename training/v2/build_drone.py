import csv,json,hashlib,os,cv2,collections,sys
sys.path.insert(0,'.'); from lib import read_labels
OUT='/home/claude/ds/merged'; REPO='/home/claude/app/HoverLogixDetect/dataset'
R=list(csv.DictReader(open(f'{REPO}/label_status.csv')))
T=json.load(open('drone/tiled.json')); P=json.load(open('/home/claude/review/seed/preds.json'))
loc=lambda f:'_'.join(f.split('__')[0].split('_')[:4])
hand_locs={loc(r['filename']) for r in R if r['source']=='hand'}
st=collections.Counter()
for r in R:
    f=r['filename']; src=r['source']
    if src not in ('hand','auto'): continue
    if src=='hand': split='test'
    elif loc(f) in hand_locs: split='val'
    else: split='val' if int(hashlib.md5(loc(f).encode()).hexdigest(),16)%100<8 else 'train'
    im=cv2.imread('/home/claude/work/data/rs/'+f); H,W=im.shape[:2]
    if src=='hand':
        labs=[(c,b) for c,b,_ in read_labels(f'{REPO}/labels/{f[:-4]}.txt')]
    else:
        t=T[f]; labs=[(q[0],[q[2]/W,q[3]/H,q[4]/W,q[5]/H]) for q in t['boxes']]
        labs+=[(5,[s[0]/W,s[1]/H,s[2]/W,s[3]/H]) for s in P[f]['sugg']]
    s=1280/max(H,W)
    if s<1: im=cv2.resize(im,(round(W*s),round(H*s)),interpolation=cv2.INTER_AREA)
    name='dims_'+f.split('__')[0]+'_'+hashlib.md5(f.encode()).hexdigest()[:8]
    os.makedirs(f'{OUT}/images/{split}',exist_ok=True); os.makedirs(f'{OUT}/labels/{split}',exist_ok=True)
    cv2.imwrite(f'{OUT}/images/{split}/{name}.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,90])
    with open(f'{OUT}/labels/{split}/{name}.txt','w') as fh:
        for c,b in labs:
            b=[min(1,max(0,v)) for v in b]
            if b[2]-b[0]<0.003 or b[3]-b[1]<0.003: continue
            fh.write(f"{c} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}\n"); st[(split,c)]+=1
    st[(split,'img')]+=1
print(sorted(st.items(),key=str))
names=['pallet','pallet_label','location_label','beam','upright','empty_location','pallet_base']
open(f'{OUT}/data.yaml','w').write("path: /content/data\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"+''.join(f"  {i}: {n}\n" for i,n in enumerate(names)))
