import json
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score, f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict, Counter
import pandas as pd

# 导入分类词典
import sys
sys.path.append('classification')
from category import category_dict

def load_data():
    """加载参考答案和预测结果"""
    # 加载参考答案
    ref_data = {}
    with open('json/ref_diagnoses.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ref_data[data['question_id']] = data['answer']
    
    # 加载预测结果
    pred_data = {}
    with open('json/qwen_diagnosis_result.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pred_data[data['question_id']] = data['text']
    
    return ref_data, pred_data

def normalize_diagnosis(diagnosis_text, category_dict):
    """将诊断结果标准化为分类"""
    if not diagnosis_text or diagnosis_text.strip() == '':
        return []
    
    # 处理多个诊断（分号或逗号分隔）
    diagnoses = []
    for sep in [';', '；', ',', '，']:
        if sep in diagnosis_text:
            diagnoses = [d.strip() for d in diagnosis_text.split(sep)]
            break
    else:
        diagnoses = [diagnosis_text.strip()]
    
    # 标准化每个诊断
    normalized = []
    for diag in diagnoses:
        if diag in category_dict:
            normalized.append(category_dict[diag])
        else:
            # 如果不在词典中，尝试部分匹配
            matched = False
            for key, value in category_dict.items():
                if key in diag or diag in key:
                    normalized.append(value)
                    matched = True
                    break
            if not matched:
                normalized.append("其他")  # 未匹配的归为其他类别
    
    return list(set(normalized))  # 去重

def prepare_multilabel_data(ref_data, pred_data, category_dict):
    """准备多标签分类的数据"""
    # 获取所有可能的标签
    all_labels = set()
    
    # 标准化数据
    normalized_ref = {}
    normalized_pred = {}
    
    for qid in ref_data:
        if qid in pred_data:
            ref_labels = normalize_diagnosis(ref_data[qid], category_dict)
            pred_labels = normalize_diagnosis(pred_data[qid], category_dict)
            
            normalized_ref[qid] = ref_labels
            normalized_pred[qid] = pred_labels
            
            all_labels.update(ref_labels)
            all_labels.update(pred_labels)
    
    all_labels = sorted(list(all_labels))
    
    # 转换为二进制矩阵
    y_true = []
    y_pred = []
    
    for qid in sorted(normalized_ref.keys()):
        true_vector = [1 if label in normalized_ref[qid] else 0 for label in all_labels]
        pred_vector = [1 if label in normalized_pred[qid] else 0 for label in all_labels]
        
        y_true.append(true_vector)
        y_pred.append(pred_vector)
    
    return np.array(y_true), np.array(y_pred), all_labels, normalized_ref, normalized_pred

def calculate_exact_match_ratio(normalized_ref, normalized_pred):
    """计算精确匹配率"""
    matches = 0
    total = 0
    
    for qid in normalized_ref:
        if qid in normalized_pred:
            ref_set = set(normalized_ref[qid])
            pred_set = set(normalized_pred[qid])
            if ref_set == pred_set:
                matches += 1
            total += 1
    
    return matches / total if total > 0 else 0

def calculate_class_specific_metrics(y_true, y_pred, all_labels):
    """计算每个类别的特定指标"""
    metrics = {}
    
    for i, label in enumerate(all_labels):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        # 计算混淆矩阵元素
        tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
        tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
        fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
        fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
        
        # 计算指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'Specificity': float(specificity),
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Sensitivity': float(recall),
            'F1-score': float(f1),
            'Support': int(np.sum(y_true_class == 1))  # 转换为Python int
        }
    
    return metrics

def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def main():
    # 加载数据
    ref_data, pred_data = load_data()
    
    # 准备多标签数据
    y_true, y_pred, all_labels, normalized_ref, normalized_pred = prepare_multilabel_data(
        ref_data, pred_data, category_dict
    )
    
    print("=== 多标签分类评估结果 ===\n")
    
    # 1. Exact Match Ratio
    exact_match = calculate_exact_match_ratio(normalized_ref, normalized_pred)
    print(f"Exact Match Ratio: {exact_match:.4f}")
    
    # 2. Hamming Loss
    hamming_loss_score = hamming_loss(y_true, y_pred)
    print(f"Hamming Loss: {hamming_loss_score:.4f}")
    
    # 3. Jaccard Score
    jaccard_micro = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    jaccard_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    jaccard_samples = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    print(f"Jaccard Score (Micro): {jaccard_micro:.4f}")
    print(f"Jaccard Score (Macro): {jaccard_macro:.4f}")
    print(f"Jaccard Score (Samples): {jaccard_samples:.4f}")
    
    # 4. F1 Scores
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Micro-F1: {f1_micro:.4f}")
    print(f"Macro-F1: {f1_macro:.4f}")
    
    # 5. 每个类别的详细指标
    class_metrics = calculate_class_specific_metrics(y_true, y_pred, all_labels)
    
    print("\n=== 各类别详细指标 ===")
    
    # 创建DataFrame用于更好的显示
    metrics_df = pd.DataFrame(class_metrics).T
    metrics_df = metrics_df.round(4)
    
    # 按支持度排序
    metrics_df = metrics_df.sort_values('Support', ascending=False)
    
    print(metrics_df.to_string())
    
    # 6. 类别分布统计
    print("\n=== 类别分布统计 ===")
    ref_counter = Counter()
    pred_counter = Counter()
    
    for labels in normalized_ref.values():
        ref_counter.update(labels)
    
    for labels in normalized_pred.values():
        pred_counter.update(labels)
    
    distribution_df = pd.DataFrame({
        'True_Count': ref_counter,
        'Pred_Count': pred_counter
    }).fillna(0).astype(int)
    
    distribution_df = distribution_df.sort_values('True_Count', ascending=False)
    print(distribution_df.to_string())
    
    # 7. 保存详细结果
    results = {
        'overall_metrics': {
            'exact_match_ratio': float(exact_match),
            'hamming_loss': float(hamming_loss_score),
            'jaccard_micro': float(jaccard_micro),
            'jaccard_macro': float(jaccard_macro),
            'f1_micro': float(f1_micro),
            'f1_macro': float(f1_macro)
        },
        'class_metrics': class_metrics,  # 已经在函数内转换了类型
        'label_distribution': {
            'true': dict(ref_counter),
            'pred': dict(pred_counter)
        }
    }
    
    # 确保所有numpy类型都被转换
    results = convert_numpy_types(results)
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到 evaluation_results.json")
    
    # 8. 输出一些样本分析
    print("\n=== 样本分析（前10个） ===")
    for i, qid in enumerate(sorted(normalized_ref.keys())[:10]):
        print(f"样本 {qid}:")
        print(f"  真实: {normalized_ref[qid]}")
        print(f"  预测: {normalized_pred[qid]}")
        print(f"  匹配: {'✓' if set(normalized_ref[qid]) == set(normalized_pred[qid]) else '✗'}")
        print()

if __name__ == "__main__":
    main()