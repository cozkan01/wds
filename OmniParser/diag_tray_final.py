import sys, cv2, numpy as np, importlib
sys.path.insert(0, 'WDS')
import icon_detector, wehgp
importlib.reload(icon_detector)
importlib.reload(wehgp)
from wehgp import process as wehgp_process

bgr = cv2.imread('clipboard_temp.png')
if bgr is None:
    print("Could not load clipboard_temp.png")
    sys.exit(1)

h, w = bgr.shape[:2]

score, mask = icon_detector._edge_map(bgr)

rx1, ry1 = w - 350, h - 80
tray_mask = mask[ry1:h, rx1:w]
tray_orig = bgr[ry1:h, rx1:w]

ys, xs = np.where(tray_mask > 0)
print(f"Edge pixels in tray region ({rx1}:{w}, {ry1}:{h}): {len(xs)}")
if len(xs) > 0:
    print(f"X range: {xs.min()+rx1} to {xs.max()+rx1},  Y range: {ys.min()+ry1} to {ys.max()+ry1}")

_, boxes, labels_keep = wehgp_process(bgr)

vis = cv2.cvtColor(tray_orig.copy(), cv2.COLOR_BGR2RGB)
for y, x in zip(ys, xs):
    vis[y, x] = (0, 255, 255)

for b in boxes:
    bx,by,bw,bh = b['x'],b['y'],b['w'],b['h']
    cx1=bx-rx1; cy1=by-ry1; cx2=cx1+bw; cy2=cy1+bh
    if cx2<0 or cy2<0 or cx1>350 or cy1>80: continue
    col = (255,80,80) if b['type']=='icon' else (80,255,80)
    cv2.rectangle(vis,(max(0,cx1),max(0,cy1)),(min(350,cx2),min(80,cy2)),col,1)
    print(f"  BOX ({b['x']},{b['y']}) {bw}x{bh} {b['type']}")

cv2.imwrite('diag_tray_final.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
big = cv2.resize(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), (350*4, 80*4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('diag_tray_final_4x.png', big)
