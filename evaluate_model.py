import json
import jieba
from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# 加载 jsonl 文件
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

# 分词函数（用于 METEOR）
def tokenize_zh(text):
    return ' '.join(jieba.lcut(text))

# 主函数
def evaluate(results_path, ref_path):
    results = load_jsonl(results_path)
    refs = load_jsonl(ref_path)

    # 构建字典：question_id -> text
    candidates = {r['question_id']: r['text'] for r in results}
    references = {r['question_id']: [r['answer']] for r in refs}

    # 检查对齐
    assert set(candidates.keys()) == set(references.keys()), "question_id 不匹配"

    # 分词处理（可选）
    candidates = {k: tokenize_zh(v) for k, v in candidates.items()}
    references = {k: [tokenize_zh(v[0])] for k, v in references.items()}

    # 构建评估格式
    gts = references
    res = {k: [v] for k, v in candidates.items()}

    # 指标列表
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    from nltk.translate.meteor_score import meteor_score
    import jieba
    
    meteor_scores = []
    for k in gts:
        ref_tokens = list(jieba.lcut(gts[k][0]))   # 参考答案分词
        pred_tokens = list(jieba.lcut(res[k][0]))  # 预测结果分词
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
    
    print(f"METEOR: {sum(meteor_scores) / len(meteor_scores):.4f}")

    print("评估结果：")
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                print(f"{m}: {s:.4f}")
        else:
            print(f"{method}: {score:.4f}")

# 运行
if __name__ == "__main__":
    evaluate("json/qwen_result_descriptions.jsonl", "json/qwen_ref_descriptions.jsonl")