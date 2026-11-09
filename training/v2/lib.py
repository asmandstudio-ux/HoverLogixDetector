import os,glob
def read_labels(p):
    out=[]
    if not os.path.exists(p): return None
    for l in open(p):
        v=l.split()
        if len(v)<5: continue
        c=int(float(v[0])); a=list(map(float,v[1:]))
        if len(a)==4:
            cx,cy,w,h=a; b=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
        else:
            xs=a[0::2]; ys=a[1::2]; b=[min(xs),min(ys),max(xs),max(ys)]
        out.append((c,b,len(a)>4))
    return out
def lbl_path(img): return img.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
