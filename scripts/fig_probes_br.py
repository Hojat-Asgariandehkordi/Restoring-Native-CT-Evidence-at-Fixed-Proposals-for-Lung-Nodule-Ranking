"""Clean the BioRender probe illustration for Fig. 2b.

Input : figures/biorender/probes.jpg  (BioRender export, 2752x1536; created in BioRender)
Output: figures/biorender/probes_clean.png  (cropped, teal clusters removed)
        figures/biorender/probes_beads.json (pixel centres of the purple sample beads)

BioRender placed the teal voxel clusters *beyond* the purple beads, but the native sources are
the voxels around each bead (Sec. 3.4). We therefore erase BioRender's teal clusters (inpainting)
and the figure script draws the teal voxels around the detected bead centres instead.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures" / "biorender" / "probes.jpg"
OUT = ROOT / "figures" / "biorender"

img = cv2.imread(str(SRC))                       # BGR
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)

# teal: hue ~ 80-95 (OpenCV 0-180), saturated
teal = (h >= 75) & (h <= 98) & (s > 90) & (v > 60)
teal = cv2.dilate(teal.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2)
clean = cv2.inpaint(img, teal, 9, cv2.INPAINT_TELEA)

# purple beads: hue ~ 128-150, saturated
purple = ((h >= 125) & (h <= 155) & (s > 70) & (v > 60)).astype(np.uint8)
purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats(purple)
keep = [i for i in range(1, n) if 1500 < stats[i, cv2.CC_STAT_AREA] < 40000]
beads = [(float(cent[i][0]), float(cent[i][1])) for i in keep]
bead_r = float(np.median([max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) / 2 for i in keep]))

# crop to the illustration (the large pale sphere) with a small margin
ys, xs = np.nonzero(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) < 245)
x0, x1, y0, y1 = xs.min() - 20, xs.max() + 20, ys.min() - 20, ys.max() + 20
crop = clean[y0:y1, x0:x1]
cv2.imwrite(str(OUT / "probes_clean.png"), crop)
centre = None
blue = ((h >= 100) & (h <= 120) & (s > 80) & (v > 60)).astype(np.uint8)
nb, labb, statsb, centb = cv2.connectedComponentsWithStats(blue)
k = 1 + int(np.argmax(statsb[1:, cv2.CC_STAT_AREA]))
centre = (float(centb[k][0] - x0), float(centb[k][1] - y0))
json.dump({"size": [int(x1 - x0), int(y1 - y0)], "centre": centre, "bead_r": bead_r,
           "beads": [(bx - x0, by - y0) for bx, by in beads]},
          open(OUT / "probes_beads.json", "w"), indent=1)
print(len(beads), "beads; crop", crop.shape, "centre", centre)
