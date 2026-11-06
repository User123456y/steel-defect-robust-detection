"""背景纹理分组诊断；相关性指标不能证明纹理干扰的因果关系。"""
import numpy as np

def background_texture(gray, boxes, minimum_valid_fraction=0.2):
    a=np.asarray(gray,dtype=np.float64)
    if a.ndim!=2: raise ValueError("Expected grayscale 2D array")
    h,w=a.shape
    bg=np.ones((h,w),dtype=bool)
    for box in boxes:
        x1,y1,x2,y2=box
        if x2<=x1 or y2<=y1: raise ValueError("Invalid GT box")
        x1,x2=max(0,int(np.floor(x1))),min(w,int(np.ceil(x2)))
        y1,y2=max(0,int(np.floor(y1))),min(h,int(np.ceil(y2)))
        bg[y1:y2,x1:x2]=False
    valid=bg[1:-1,1:-1]&bg[:-2,1:-1]&bg[2:,1:-1]&bg[1:-1,:-2]&bg[1:-1,2:]
    count=int(valid.sum())
    fraction=count/(h*w)
    dx=(a[1:-1,2:]-a[1:-1,:-2])/2
    dy=(a[2:,1:-1]-a[:-2,1:-1])/2
    eligible=count>0 and fraction>=minimum_valid_fraction
    score=float(np.sqrt(np.mean((dx*dx+dy*dy)[valid]))/255) if eligible else None
    return {"score":score,"valid_pixels":count,"image_pixels":h*w,"valid_fraction":fraction,"eligible_background":eligible}

def fit_thresholds(rows, categories, quantile=.75, minimum_count=30):
    result={}
    for c in categories:
        vals=[r["score"] for r in rows if r["split"]=="train" and r["acquisition_class_id"]==c and r["score"] is not None]
        result[str(c)]={"eligible_train_images":len(vals),"threshold":float(np.quantile(vals,quantile,method="linear")) if len(vals)>=minimum_count else None}
    return result

def assign_group(row,thresholds):
    t=thresholds[str(row["acquisition_class_id"])]["threshold"]
    if row["score"] is None or t is None: return "undefined"
    return "high" if row["score"]>=t else "nonhigh"

def validate_predictions(data, records, manifest_hash):
    md=data["metadata"]
    if md["split"]!="val" or md["manifest_sha256"]!=manifest_hash: raise ValueError("Wrong split or manifest")
    if md["protocol"]["confidence"]!=.25 or md["protocol"]["match_iou"]!=.5: raise ValueError("Wrong operating point")
    expected={r["output_image"]:r for r in records if r["split"]=="val"}
    ims=data["images"]
    if len(ims)!=len(expected) or {i["output_image"] for i in ims}!=set(expected): raise ValueError("Image membership mismatch")
    for im in ims:
        r=expected[im["output_image"]]
        if im["image_sha256"]!=r["image_sha256"]: raise ValueError("Image hash mismatch")
        gt=im["ground_truth"]
        if len(gt)!=len(r["boxes"]): raise ValueError("GT count mismatch")
        for g,b in zip(gt,r["boxes"]):
            if g["class_id"]!=b[0] or not np.allclose(g["xyxy"],b[1:],rtol=0,atol=1e-7): raise ValueError("GT mismatch")
