#!/bin/bash

python convert_for_bleu_rouge.py 

python convert_for_coco.py

python - <<'PY'                                               
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

coco = COCO('/json/llava_results/ref_coco.json')
cocoRes = coco.loadRes('/json/llava_results/pred_list.json')

cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.evaluate()

for k, v in cocoEval.eval.items():
    print(f"{k}: {v:.3f}")
PY