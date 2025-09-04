import argparse
import json
import os

def read_lines(p):
    with open(p, encoding='utf-8') as f:
        return [ln.strip() for ln in f]

def main():
    base = f"./json/llava_results"

    ref_lines  = read_lines(f"{base}/ref.txt")
    pred_lines = read_lines(f"{base}/pred.txt")

    # 1) 参考文件：完整 COCO 字典格式
    ref_coco = {
        "info": {},
        "licenses": [],
        "images": [{"id": idx} for idx in range(len(ref_lines))],
        "annotations": [
            {"id": idx, "image_id": idx, "caption": cap}
            for idx, cap in enumerate(ref_lines)
        ]
    }

    # 2) 预测文件：loadRes 需要的纯列表
    pred_list = [
        {"image_id": idx, "caption": cap}
        for idx, cap in enumerate(pred_lines)
    ]

    ref_path  = f"{base}/ref_coco.json"
    pred_path = f"{base}/pred_list.json"

    with open(ref_path,  'w', encoding='utf-8') as f:
        json.dump(ref_coco,  f, ensure_ascii=False, indent=2)
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(pred_list, f, ensure_ascii=False, indent=2)

    print(f"已生成\n  {ref_path}\n  {pred_path}")

if __name__ == '__main__':
    main()