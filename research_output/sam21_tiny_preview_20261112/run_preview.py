from pathlib import Path
import sys, csv, json, time
import cv2
import numpy as np
import torch
sys.path.insert(0, str(Path('phase2_reorientation_validation/tools').resolve()))
from sam21_hsv_prompt_pilot import hsv_prompt, preview
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

out = Path(__file__).resolve().parent
root = Path('research_output/perception_validation/benchmark_v0')
checkpoint = '/home/nguyenchianh7777/sam2/checkpoints/sam2.1_hiera_tiny.pt'
torch.set_num_threads(4)
predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', checkpoint, device='cpu', apply_postprocessing=False))
with (root/'annotation_manifest.csv').open() as f:
    rows = list(csv.DictReader(f))
rows = [r for r in rows if r['camera']=='robot_0_head' and int(r['frame']) in [600,1250,1500,1600,1950,2200,2600]]
records=[]
writer=cv2.VideoWriter(str(out/'sam21_tiny_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(1152,460))
if not writer.isOpened():
    raise RuntimeError('Cannot open MP4 writer')
for r in rows:
    rgb=cv2.cvtColor(cv2.imread(str(root/r['rgb'])),cv2.COLOR_BGR2RGB)
    hsv=cv2.imread(str(root/r['hsv_prediction']),0)>127
    prompt=hsv_prompt(hsv,8)
    if prompt is None: continue
    box,point=prompt
    start=time.perf_counter()
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks,scores,_=predictor.predict(box=box,point_coords=point,point_labels=np.array([1],dtype=np.int32),multimask_output=False)
    mask=masks[0].astype(bool)
    frame=cv2.cvtColor(preview(rgb,hsv,mask,box,point,r['sample_id'],float(scores[0])),cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out/(r['sample_id']+'.png')),frame)
    cv2.imwrite(str(out/(r['sample_id']+'_mask.png')),mask.astype(np.uint8)*255)
    for _ in range(4): writer.write(frame)
    record={'sample':r['sample_id'],'seconds':round(time.perf_counter()-start,2),'predicted_iou':float(scores[0]),'mask_pixels':int(mask.sum())}
    records.append(record)
    print(record,flush=True)
writer.release()
(out/'summary.json').write_text(json.dumps({'device':'cpu','checkpoint':checkpoint,'note':'HSV prompted predictions; video is a slideshow of sampled frames, not continuous tracking','results':records},indent=2))
if not records: raise RuntimeError('No frames processed')
print('Output:',out,flush=True)
