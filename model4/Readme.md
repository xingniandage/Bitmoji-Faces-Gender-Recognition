# Readme

---

## 1. 使用模型

模型为DenseNet169。

## 2.训练方法

1. 使用keras预训练模型，加载imagenet预训练权值。
2. 定义模型顶部全连接层，冻结预训练模型，进行小epoch_size的预训练，学习二分类方法。
3. 解冻部分模型，进行大epoch_size训练，实现模型调优。

## 3.训练策略

1. 使用EarlyStopping，在模型劣化epoch超过阈值之后回退到最好版本。
2. 每个epoch将数据random_shuffle一次，增强模型鲁棒性。
3. 预训练时使用较大学习率，快速学习分类方法；训练时使用小100倍的学习率，缓速进行模型调优，不丢失预训练特征。

## 4.训练结果

在kaggle的比赛上达到了0.98的正确率，具体训练图表可以使用 `tensorboard --logdir=".\result_v7\results\bitmoji\logs\train"`查看。