import json, argparse, os, re
# from datasets import load_metric
import jieba  # 中文用 jieba 分词，英文可去掉


def clean(text: str) -> str:
    # 去掉所有换行和多余空格
    text = re.sub(r'\s+', ' ', text)
    # 提取“描述：”到“诊断：”之间的内容
    m = re.search(r'描述[:：]\s*(.*?)(?:\s*诊断[:：]|$)', text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()          # 兜底

base = f"./json"
pred_file = f"{base}/llava_result.jsonl"
ref_file  = f"{base}/llava_ref_answers.jsonl"
out_dir   = f"{base}/llava_results"
os.makedirs(out_dir, exist_ok=True)

# 读
pred = {d['question_id']: clean(d['text'])   for d in map(json.loads, open(pred_file))}
ref  = {d['question_id']: clean(d['answer']) for d in map(json.loads, open(ref_file))}

# 生成预测文件和参考文件（一行对一行，顺序对齐）
with open(f"{out_dir}/pred.txt", 'w', encoding='utf-8') as fp, \
     open(f"{out_dir}/ref.txt",  'w', encoding='utf-8') as fr:
    for qid in sorted(ref):
        fp.write(' '.join(jieba.lcut(pred.get(qid, ''))) + '\n')
        fr.write(' '.join(jieba.lcut(ref[qid])) + '\n')

print("已生成 pred.txt / ref.txt，可直接用 sacrebleu / rouge-score 计算。")
