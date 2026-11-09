import glob,os,json,numpy as np,cv2,onnxruntime as ort,math
S=960; TH={0:.45,1:.35,2:.35,3:.35,4:.35}
sess=ort.InferenceSession('/home/claude/app/HoverLogixDetect/models/hoverlogix_v1_960.onnx',providers=['CPUExecutionProvider'])
iname=sess.get_inputs()[0].name
def det(img,th):
    H,W=img.shape[:2]; r=min(S/W,S/H); nw,nh=int(W*r+.5),int(H*r+.5); px,py=(S-nw)//2,(S-nh)//2
    c=np.full((S,S,3),114,np.uint8); c[py:py+nh,px:px+nw]=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
    y=sess.run(None,{iname:cv2.cvtColor(c,cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1)[None]/255})[0][0]
    sc=y[4:]; best=sc.argmax(0); bs=sc.max(0); out=[]
    for a in np.nonzero(bs>min(th.values()))[0]:
        cl=int(best[a]); s=float(bs[a])
        if s<th[cl]: continue
        cx,cy,w,h=[float(v) for v in y[:4,a]]
        b=[(cx-w/2-px)/r,(cy-h/2-py)/r,(cx+w/2-px)/r,(cy+h/2-py)/r]
        out.append([cl,s,max(0,b[0]),max(0,b[1]),min(W,b[2]),min(H,b[3])])
    return out
def iou(a,b):
    iw=min(a[4],b[4])-max(a[2],b[2]); ih=min(a[5],b[5])-max(a[3],b[3])
    if iw<=0 or ih<=0: return 0
    I=iw*ih; return I/((a[4]-a[2])*(a[5]-a[3])+(b[4]-b[2])*(b[5]-b[3])-I)
def nms(L):
    keep=[]
    for cl in range(5):
        Q=sorted([q for q in L if q[0]==cl],key=lambda q:-q[1])
        while Q:
            b=Q.pop(0); keep.append(b); Q=[q for q in Q if iou(b,q)<=0.5 and not (cl in (1,2) and contain(b,q))]
    return keep
def contain(a,b):  # small box mostly inside big one of same class -> duplicate
    iw=min(a[4],b[4])-max(a[2],b[2]); ih=min(a[5],b[5])-max(a[3],b[3])
    if iw<=0 or ih<=0: return False
    return iw*ih > 0.8*min((a[4]-a[2])*(a[5]-a[3]),(b[4]-b[2])*(b[5]-b[3]))
res={}
fs=sorted(glob.glob('/home/claude/work/data/rs/*.jpg'))
for k,f in enumerate(fs):
    img=cv2.imread(f); H,W=img.shape[:2]
    L=det(img,TH); full=len(L)
    # two overlapping tiles along the long side
    if H>=W: tiles=[(0,0,W,int(H*.58)),(0,int(H*.42),W,H)]
    else: tiles=[(0,0,int(W*.58),H),(int(W*.42),0,W,H)]
    for (x0,y0,x1,y1) in tiles:
        for q in det(img[y0:y1,x0:x1],{0:.99,1:.4,2:.4,3:.45,4:.99}):
            q=[q[0],q[1],q[2]+x0,q[3]+y0,q[4]+x0,q[5]+y0]
            # drop boxes cut by the tile's inner edge
            if (y0>0 and q[3]<=y0+2) or (y1<H and q[5]>=y1-2) or (x0>0 and q[2]<=x0+2) or (x1<W and q[4]>=x1-2): continue
            L.append(q)
    K=nms(L); res[os.path.basename(f)]={'w':W,'h':H,'boxes':[[q[0],round(q[1],3)]+[round(v,1) for v in q[2:]] for q in K],'full':full}
    if k%50==0: print(k,flush=True)
json.dump(res,open('/home/claude/ds/drone/tiled.json','w')); print('done',len(res))
