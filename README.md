# Slit Lamp Image Report Generation Test - LLaVA vs Qwen2.5-VL

This project evaluates the performance of large vision-language models on ophthalmic slit lamp image description generation tasks, primarily comparing LLaVA and Qwen2.5-VL models.

## Evaluation Methods

### LLaVA Model Evaluation

#### 1. Description Generation Quality Assessment
```bash
bash eval_llava.sh
```

Supported evaluation metrics:
- **BLEU**: N-gram based translation quality assessment
- **ROUGE**: Summary quality evaluation metric
- **METEOR**: Machine translation evaluation considering synonyms
- **CIDEr**: Image description evaluation metric
- **SPICE**: Scene graph-based image description evaluation

#### 2. Disease Classification Evaluation
```bash
python multi_classification_llava.py
```

Evaluates:
- Multi-class classification accuracy
- Precision, Recall, F1-score per disease category
- Confusion matrix analysis

### Qwen2.5-VL Model Evaluation

#### 1. Description Generation Quality Assessment
Use multiple metrics to evaluate generated text quality:

```bash
python evaluate_model.py
```


#### 2. Disease Classification Evaluation
```bash
python multi_classification.py
```

Provides:
- Classification performance metrics
- Disease-specific accuracy analysis
- Statistical significance testing


## Results

The evaluation outputs include:

### Quantitative Metrics
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram precision scores
- **ROUGE-L**: Longest common subsequence based recall
- **METEOR**: Harmonic mean of precision and recall
- **CIDEr**: Consensus-based evaluation
- **SPICE**: Semantic content evaluation

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged scores
