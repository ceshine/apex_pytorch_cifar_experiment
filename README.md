# APEX PyTorch Cifar-10 Experiments

## Results

Batch size 128 | [Wide Resnet](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py) | 10 epochs | Linear LR scheduler with Warmup

### Google Colab (T4/K80)

Notebook snapshots stored in [colab_snapshots](colab_snapshots/) subfolder.

| Level                | GPU | Time        | GPU Memory | Validation Accuracy |
|----------------------|-----|-------------|------------|---------------------|
| O0 (Pure FP32)       | T4  | 57min 17s   | 7657 MB    | 86.75%              |
| O1 (Mixed Precision) | T4  | 31min 14s   | 4433 MB    | 85.25%              |
| O2 (Mixed Precision) | T4  | 29min 31s   | 4383 MB    | 88.90%              |
| O3 (Pure FP16)       | T4  | N/A         | 4347 MB    | Not Converged.      |
| O0 (Pure FP32)       | K80 | 1h 43min 7s | 6786 MB    | 88.44%              |

### Kaggle (P100)

Kaggle Kernel used: [APEX Experiment - Cifar 10](https://www.kaggle.com/ceshine/apex-experiment-cifar-10).

| Level                | Time      | GPU Memory | Validation Accuracy |
|----------------------|-----------|------------|---------------------|
| [O0 (Pure FP32)](https://www.kaggle.com/ceshine/apex-experiment-cifar-10?scriptVersionId=15544605)       | 47min 01s | 5677 MB    | 87.49%              |
| [O1 (Mixed Precision)](https://www.kaggle.com/ceshine/apex-experiment-cifar-10?scriptVersionId=15544647) | 47min 34s | 6283 MB    | 88.51%              |
| [O2 (Mixed Precision)](https://www.kaggle.com/ceshine/apex-experiment-cifar-10?scriptVersionId=15556913) | 45min 34s | 5665 MB    | 87.74%              |