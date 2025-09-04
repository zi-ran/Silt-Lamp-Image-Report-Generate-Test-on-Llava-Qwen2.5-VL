import json
import re
from collections import Counter
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score
)
from category import category_dict

def extract_diagnoses_from_text(text):
    """从文本中提取*诊断*:后的诊断内容，适配新格式（带空格）"""
    # 匹配 * 诊断 * ： 或 *诊断*： 格式
    pattern = r'\*\s*诊断\s*\*\s*[：:]\s*([^*]+?)(?:\*|$)'
    match = re.search(pattern, text)
    if match:
        diagnoses_text = match.group(1).strip()
        # 按逗号、分号或换行分割多个诊断
        diagnoses = re.split(r'[,，;；\n]+', diagnoses_text)
        return [d.strip() for d in diagnoses if d.strip()]
    return []

def map_to_categories(diagnoses, category_dict):
    """将诊断映射到标准类别"""
    categories = set()
    for diagnosis in diagnoses:
        # 去除空格进行匹配
        diagnosis_clean = diagnosis.replace(' ', '')
        
        # 首先尝试直接匹配
        if diagnosis_clean in category_dict:
            categories.add(category_dict[diagnosis_clean])
        elif diagnosis in category_dict:
            categories.add(category_dict[diagnosis])
        else:
            # 尝试部分匹配
            found = False
            for key, value in category_dict.items():
                key_clean = key.replace(' ', '')
                if (key_clean in diagnosis_clean or diagnosis_clean in key_clean or
                    key in diagnosis or diagnosis in key):
                    categories.add(value)
                    found = True
                    break
            if not found:
                categories.add("其他")
    return list(categories)

def load_pred_list_data(file_path):
    """加载新格式的JSON文件数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_ref_data(file_path):
    """加载参考答案JSONL文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_labels_from_pred_data(data):
    """从预测数据中提取标签"""
    all_labels = []
    for item in data:
        caption = item.get('caption', '')
        diagnoses = extract_diagnoses_from_text(caption)
        categories = map_to_categories(diagnoses, category_dict)
        all_labels.append(categories)
    return all_labels

def extract_labels_from_ref_data(data):
    """从参考数据中提取标签"""
    all_labels = []
    for item in data:
        answer = item.get('answer', '')
        diagnoses = extract_diagnoses_from_text(answer)
        categories = map_to_categories(diagnoses, category_dict)
        all_labels.append(categories)
    return all_labels

def get_all_unique_labels(pred_labels, true_labels):
    """获取所有唯一的标签"""
    all_labels = set()
    for labels in pred_labels + true_labels:
        all_labels.update(labels)
    return sorted(list(all_labels))

def labels_to_binary_matrix(labels_list, all_labels):
    """将标签列表转换为二进制矩阵"""
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    binary_matrix = np.zeros((len(labels_list), len(all_labels)))
    for i, labels in enumerate(labels_list):
        for label in labels:
            if label in label_to_idx:
                binary_matrix[i, label_to_idx[label]] = 1
    return binary_matrix

def calculate_multilabel_accuracy(y_true, y_pred):
    """
    计算多标签分类的准确率
    准确率 = 完全匹配的样本数 / 总样本数
    """
    return np.all(y_true == y_pred, axis=1).mean()

def calculate_subset_accuracy(y_true, y_pred):
    """
    计算子集准确率（与完全匹配相同）
    """
    return calculate_multilabel_accuracy(y_true, y_pred)

def calculate_sample_wise_accuracy(y_true, y_pred):
    """
    计算样本级准确率
    每个样本的准确率 = (TP + TN) / (TP + TN + FP + FN)
    """
    sample_accuracies = []
    for i in range(y_true.shape[0]):
        tp = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
        tn = np.sum((y_true[i] == 0) & (y_pred[i] == 0))
        fp = np.sum((y_true[i] == 0) & (y_pred[i] == 1))
        fn = np.sum((y_true[i] == 1) & (y_pred[i] == 0))
        
        total = tp + tn + fp + fn
        if total > 0:
            accuracy = (tp + tn) / total
        else:
            accuracy = 1.0  # 如果没有标签，认为完全正确
        sample_accuracies.append(accuracy)
    
    return np.mean(sample_accuracies)

def calculate_per_class_accuracy(y_true, y_pred):
    """
    计算每个类别的准确率
    对于每个类别: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    accuracies = []
    for i in range(y_true.shape[1]):  # 对每个类别
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        
        total = tp + tn + fp + fn
        if total > 0:
            accuracy = (tp + tn) / total
        else:
            accuracy = 1.0  # 如果没有样本，准确率为1
        accuracies.append(accuracy)
    
    return np.array(accuracies)

def calculate_multilabel_specificity(y_true, y_pred):
    """
    计算多标签分类的特异性
    特异性 = TN / (TN + FP)
    """
    specificities = []
    for i in range(y_true.shape[1]):  # 对每个类别
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 1.0  # 如果没有负样本，特异性为1
        specificities.append(specificity)
    
    return np.array(specificities)

def calculate_per_sample_specificity(y_true, y_pred):
    """
    计算每个样本的特异性，然后取平均
    """
    sample_specificities = []
    for i in range(y_true.shape[0]):
        tn = np.sum((y_true[i] == 0) & (y_pred[i] == 0))
        fp = np.sum((y_true[i] == 0) & (y_pred[i] == 1))
        
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 1.0
        sample_specificities.append(specificity)
    
    return np.mean(sample_specificities)

def main():
    print("=== 新模型多标签分类评估报告 ===")
    
    # 加载数据
    print("加载预测数据...")
    pred_data = load_pred_list_data("json/llava_result_diagnosis.json")
    print("加载参考数据...")
    true_data = load_ref_data("json/llava_ref_answers.jsonl")

    # 提取标签
    print("提取预测标签...")
    pred_labels = extract_labels_from_pred_data(pred_data)
    print("提取真实标签...")
    true_labels = extract_labels_from_ref_data(true_data)
    
    # 确保数据量一致
    min_length = min(len(pred_labels), len(true_labels))
    pred_labels = pred_labels[:min_length]
    true_labels = true_labels[:min_length]
    
    print(f"数据样本数: {min_length}")
    print(f"预测数据总量: {len(pred_data)}")
    print(f"参考数据总量: {len(true_data)}")
    
    # 获取所有标签
    all_labels = get_all_unique_labels(pred_labels, true_labels)
    print(f"总类别数: {len(all_labels)}")
    
    # 标签分布统计
    pred_counter = Counter()
    true_counter = Counter()
    for labels in pred_labels:
        pred_counter.update(labels)
    for labels in true_labels:
        true_counter.update(labels)
    
    print("\n=== 标签分布统计 ===")
    print(f"{'类别':<25} {'预测数':<8} {'真实数':<8} {'预测率':<8} {'真实率':<8}")
    print("-" * 70)
    
    total_pred = sum(pred_counter.values())
    total_true = sum(true_counter.values())
    
    for label in sorted(all_labels):
        pred_count = pred_counter.get(label, 0)
        true_count = true_counter.get(label, 0)
        pred_rate = pred_count / total_pred if total_pred > 0 else 0
        true_rate = true_count / total_true if total_true > 0 else 0
        if true_count > 0 or pred_count > 0:
            print(f"{label:<25} {pred_count:<8} {true_count:<8} {pred_rate:<8.3f} {true_rate:<8.3f}")
    
    # 转换为二进制矩阵
    y_pred = labels_to_binary_matrix(pred_labels, all_labels)
    y_true = labels_to_binary_matrix(true_labels, all_labels)
    
    # 计算基础指标
    exact_match = np.all(y_true == y_pred, axis=1).mean()
    hamming = hamming_loss(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average='samples')
    
    # 计算准确率指标
    subset_accuracy = calculate_subset_accuracy(y_true, y_pred)
    sample_wise_accuracy = calculate_sample_wise_accuracy(y_true, y_pred)
    per_class_accuracy = calculate_per_class_accuracy(y_true, y_pred)
    
    # 计算特异性指标
    per_class_specificity = calculate_multilabel_specificity(y_true, y_pred)
    micro_specificity = calculate_per_sample_specificity(y_true, y_pred)
    macro_specificity = np.mean(per_class_specificity)
    
    # 计算准确率的平均值
    macro_accuracy = np.mean(per_class_accuracy)
    
    # 计算支持度加权的特异性和准确率
    support = y_true.sum(axis=0)
    total_support = np.sum(support)
    if total_support > 0:
        weighted_specificity = np.average(per_class_specificity, weights=support)
        weighted_accuracy = np.average(per_class_accuracy, weights=support)
    else:
        weighted_specificity = macro_specificity
        weighted_accuracy = macro_accuracy
    
    # 计算精确率、召回率、F1分数
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n=== 整体指标 ===")
    print(f"完全匹配比例 (Exact Match): {exact_match:.4f}")
    print(f"子集准确率 (Subset Accuracy): {subset_accuracy:.4f}")
    print(f"样本级准确率 (Sample-wise Accuracy): {sample_wise_accuracy:.4f}")
    print(f"汉明损失 (Hamming Loss): {hamming:.4f}")
    print(f"Jaccard相似度: {jaccard:.4f}")
    print(f"微平均 F1: {micro_f1:.4f}")
    print(f"宏平均 F1: {macro_f1:.4f}")
    print(f"加权平均 F1: {weighted_f1:.4f}")
    
    print("\n=== 准确率和特异性指标 ===")
    print(f"宏平均准确率 (Macro Accuracy): {macro_accuracy:.4f}")
    print(f"加权平均准确率 (Weighted Accuracy): {weighted_accuracy:.4f}")
    print(f"微平均特异性 (Micro Specificity): {micro_specificity:.4f}")
    print(f"宏平均特异性 (Macro Specificity): {macro_specificity:.4f}")
    print(f"加权平均特异性 (Weighted Specificity): {weighted_specificity:.4f}")
    
    print("\n=== 详细平均指标 ===")
    print(f"{'平均方式':<15} {'精确率':<10} {'召回率':<10} {'准确率':<10} {'特异性':<10} {'F1分数':<10}")
    print("-" * 75)
    print(f"{'微平均':<15} {micro_precision:<10.4f} {micro_recall:<10.4f} {sample_wise_accuracy:<10.4f} {micro_specificity:<10.4f} {micro_f1:<10.4f}")
    print(f"{'宏平均':<15} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_accuracy:<10.4f} {macro_specificity:<10.4f} {macro_f1:<10.4f}")
    print(f"{'加权平均':<15} {weighted_precision:<10.4f} {weighted_recall:<10.4f} {weighted_accuracy:<10.4f} {weighted_specificity:<10.4f} {weighted_f1:<10.4f}")
    
    # 每个类别的指标
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    print("\n=== 各类别指标 ===")
    print(f"{'类别':<25} {'特异性':<8} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1':<8} {'支持数':<8}")
    print("-" * 95)
    
    # 按F1分数排序显示
    class_performance = []
    for i, label in enumerate(all_labels):
        if support[i] > 0:  # 只显示有真实样本的类别
            class_performance.append((
                label, 
                per_class_specificity[i],
                per_class_accuracy[i],
                per_class_precision[i], 
                per_class_recall[i], 
                
                
                per_class_f1[i], 
                int(support[i])
            ))
    
    class_performance.sort(key=lambda x: x[5], reverse=True)  # 按F1分数降序排列
    
    for label, precision, recall, accuracy, specificity, f1, sup in class_performance:
        print(f"{label:<25} {precision:<8.4f} {recall:<8.4f} {accuracy:<8.4f} {specificity:<8.4f} {f1:<8.4f} {sup:<8}")
    
    print("\n=== 性能分析 ===")
    good_classes = [x for x in class_performance if x[5] > 0.5]
    medium_classes = [x for x in class_performance if 0.3 <= x[5] <= 0.5]
    poor_classes = [x for x in class_performance if x[5] == 0 and x[6] > 2]
    
    if good_classes:
        print("表现优秀的类别 (F1 > 0.5):")
        for label, _, _, accuracy, _, f1, sup in good_classes:
            print(f"  {label}: F1={f1:.4f}, 准确率={accuracy:.4f}, 支持数={sup}")
    
    if medium_classes:
        print("表现中等的类别 (0.3 ≤ F1 ≤ 0.5):")
        for label, _, _, accuracy, _, f1, sup in medium_classes:
            print(f"  {label}: F1={f1:.4f}, 准确率={accuracy:.4f}, 支持数={sup}")
    
    if poor_classes:
        print("需要改进的类别 (F1=0, 支持数>2):")
        for label, _, _, accuracy, _, f1, sup in poor_classes:
            print(f"  {label}: 准确率={accuracy:.4f}, 支持数={sup}")
    
    # 高特异性但低召回率的类别
    high_spec_low_recall = [x for x in class_performance if x[4] > 0.9 and x[2] < 0.3]
    if high_spec_low_recall:
        print("高特异性但低召回率的类别 (特异性>0.9, 召回率<0.3):")
        for label, _, recall, accuracy, specificity, _, sup in high_spec_low_recall:
            print(f"  {label}: 准确率={accuracy:.4f}, 特异性={specificity:.4f}, 召回率={recall:.4f}, 支持数={sup}")
    
    # 预测错误分析
    print("\n=== 预测错误样本分析 (前10个) ===")
    error_count = 0
    for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
        if set(pred) != set(true) and error_count < 10:
            error_count += 1
            print(f"样本 {i+1}:")
            print(f"  预测: {pred}")
            print(f"  真实: {true}")
            missing = set(true) - set(pred)
            extra = set(pred) - set(true)
            if missing:
                print(f"  漏诊: {list(missing)}")
            if extra:
                print(f"  误诊: {list(extra)}")
            print()
    
    # 保存结果
    results = {
        "model": "new_model_pred_list",
        "dataset_info": {
            "pred_samples": len(pred_data),
            "ref_samples": len(true_data),
            "evaluated_samples": min_length,
            "total_categories": len(all_labels)
        },
        "overall_metrics": {
            "exact_match_ratio": float(exact_match),
            "subset_accuracy": float(subset_accuracy),
            "sample_wise_accuracy": float(sample_wise_accuracy),
            "hamming_loss": float(hamming),
            "jaccard_score": float(jaccard),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1)
        },
        "accuracy_metrics": {
            "macro_accuracy": float(macro_accuracy),
            "weighted_accuracy": float(weighted_accuracy)
        },
        "specificity_metrics": {
            "micro_specificity": float(micro_specificity),
            "macro_specificity": float(macro_specificity),
            "weighted_specificity": float(weighted_specificity)
        },
        "detailed_metrics": {
            "micro": {
                "precision": float(micro_precision), 
                "recall": float(micro_recall), 
                "accuracy": float(sample_wise_accuracy),
                "specificity": float(micro_specificity),
                "f1": float(micro_f1)
            },
            "macro": {
                "precision": float(macro_precision), 
                "recall": float(macro_recall), 
                "accuracy": float(macro_accuracy),
                "specificity": float(macro_specificity),
                "f1": float(macro_f1)
            },
            "weighted": {
                "precision": float(weighted_precision), 
                "recall": float(weighted_recall), 
                "accuracy": float(weighted_accuracy),
                "specificity": float(weighted_specificity),
                "f1": float(weighted_f1)
            }
        },
        "per_class_metrics": {
            "labels": all_labels,
            "precision": per_class_precision.tolist(),
            "recall": per_class_recall.tolist(),
            "accuracy": per_class_accuracy.tolist(),
            "specificity": per_class_specificity.tolist(),
            "f1": per_class_f1.tolist(),
            "support": support.tolist()
        },
        "label_distribution": {
            "predicted": dict(pred_counter),
            "true": dict(true_counter)
        }
    }
    
    output_file = "new_model_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存到: {output_file}")
    
    # 诊断提取样例检查
    print("\n=== 诊断提取样例检查 (前5个) ===")
    for i in range(min(5, len(pred_data))):
        caption = pred_data[i].get('caption', '')
        extracted = extract_diagnoses_from_text(caption)
        mapped = map_to_categories(extracted, category_dict)
        print(f"样本 {i+1}:")
        print(f"  原文: {caption[:100]}...")
        print(f"  提取: {extracted}")
        print(f"  映射: {mapped}")
        print()

if __name__ == "__main__":
    main()