# LRA Listops Benchmark

## A Practical Study

**Instructor:** Hamza Alami

**Students:**
- Yassir Fri
- Zineb Abercha
- Anass Kemmoune

---

## Table of Contents
1. [Introduction](#introduction)
    - [Project Context](#project-context)
    - [Objectives](#objectives)
2. [Task and Dataset Description](#task-and-dataset-description)
    - [Overview of the Listops Dataset](#overview-of-the-listops-dataset)
    - [Listops: Task Description](#listops-task-description)
3. [Models](#models)
    - [Mega and Longformer](#mega-and-longformer)
        - [MEGA Model (Moving Average Equipped Gated Attention)](#mega-model-moving-average-equipped-gated-attention)
        - [Longformer Model](#longformer-model)
        - [Training Environment](#training-environment)
        - [Performance Metrics](#performance-metrics)
        - [Training Configuration](#training-configuration)
        - [Comparative Analysis](#comparative-analysis)
        - [Discussion and Conclusion](#discussion-and-conclusion)
        - [Conclusion](#conclusion)
    - [BigBird Model](#bigbird-model)
        - [Training Environment](#training-environment-1)
        - [Results and Analysis (BigBird)](#results-and-analysis-bigbird)
            - [Performance Metrics](#performance-metrics-1)
            - [Training Configuration and Metrics](#training-configuration-and-metrics)
            - [Challenges and Limitations](#challenges-and-limitations)
            - [Insights and Discussion](#insights-and-discussion)
    - [Comparative RNN Implementation](#comparative-rnn-implementation)
4. [References](#references)

---

## Introduction

### Project Context
This study focuses on benchmarking multiple AI models on the Listops dataset. The Listops task evaluates a model's ability to handle hierarchical reasoning in long-context scenarios, making it an essential benchmark for understanding the limitations and strengths of different architectures.

### Objectives
- Assess the performance of state-of-the-art models on the Listops dataset.
- Investigate how sequence length affects the hierarchical reasoning capabilities of models.
- Establish baselines for long-context reasoning tasks.

---

## Task and Dataset Description

### Overview of the Listops Dataset
The Listops dataset, proposed by Nangia and Bowman (2018), involves sequences with hierarchical structures and logical operators such as `MAX`, `MEAN`, `MEDIAN`, and `SUM MOD`. Each sequence requires parsing its structure to predict the correct output.

**Data Generation:**
The experimental data was generated using two approaches:

**Base Dataset:**
- Generated using the original script from the ListOps GitHub repository.
- Produced three TSV files:
  - Training set
  - Test set
  - Validation set

**Depth-20 Dataset:**
- Additional dataset with tree depth (depth=20)

### Listops: Task Description
For this study, we used an extended version of Listops with sequence lengths up to 2K tokens to test the models' capabilities in long-context scenarios. An example sequence is:
```text
[MAX 4 3 [MIN 2 3] 1 0 [MEDIAN 1 5 8 9, 2]] → 5
```




This task involves ten-way classification, posing significant challenges for neural models.

---

## Models

### Mega and Longformer

#### MEGA Model (Moving Average Equipped Gated Attention)


The MEGA model uses gated attention with moving averages for long-context reasoning. Training was conducted using Kaggle's P100 GPU environment (16 GB of vRAM). The model was trained on the Listops dataset using:

- **Optimizer:** AdamW
- **Learning Rate Scheduler:** Cosine Annealing
- **Batch Size:** 128 with sequence length (1024), 2 with sequence length (8192)

**Training Approach:**
1. **Pretrained weights initialization:** 
   The training process began with the initialization of the model using pretrained weights from the `MegaModel` obtained from Hugging Face [https://huggingface.co/megamodel](https://huggingface.co/megamodel). This allowed the model to leverage prior knowledge from a large-scale dataset, significantly improving the starting point for training and reducing the number of epochs required for convergence.
2. **Loss function:** 
   The loss function used was standard cross-entropy loss. This function measures the difference between the predicted probability distribution and the true label distribution, enabling the model to minimize classification errors effectively.
3. **Custom loss weighting:** 
   Additional weighting was applied to reflect hierarchical relationships between classes. Misclassifications were penalized based on the semantic proximity of class labels.
4. **Training duration and early stopping:** 
   The model was trained for a total of 10 epochs with early stopping at epoch 9 based on validation accuracy.
5. **Learning rate schedule:** 
   A cosine annealing scheduler was employed to adjust the learning rate dynamically during training.

#### Longformer Model


Baseline experiments were conducted using the Longformer model. Although it demonstrated reasonable performance, its ability to handle extremely long sequences was limited compared to MEGA.

#### Training Environment
The models were trained on Kaggle with the following specifications:
- **GPU:** NVIDIA Tesla P100
- **Runtime:** Python 3.8
- **Libraries:** PyTorch, Transformers

#### Performance Metrics
The models were evaluated using classification accuracy on the Listops dataset. MEGA demonstrated superior performance compared to Longformer on sequences of both 1K and 2K tokens, achieving higher accuracy while showcasing its efficiency with smaller batch sizes and shorter training durations.

#### Training Configuration
The Longformer was trained for 3 epochs with a batch size of 2 and a maximum sequence length of 4096. Each epoch took approximately 50 minutes to complete. MEGA, on the other hand, was trained for 10 epochs with a batch size of 128 and a maximum sequence length of 1024, with each epoch taking about 9 minutes.

**Training Configuration and Validation Scores:**

| Model        | Epochs           | Max Seq Length | Batch Size | Final Validation Score |
|--------------|------------------|-----------------|------------|------------------------|
| Longformer   | 3 (50 min/epoch) | 4096            | 2          | 0.18                   |
| MEGA         | 10 (9 min/epoch) | 1024            | 128        | 0.52                   |

#### Comparative Analysis
A comparison of the classification accuracy for Longformer and MEGA on 1K and 8K token sequences reveals that MEGA consistently achieved better performance.

| Model        | Accuracy (8K Tokens) | Accuracy (1K Tokens) |
|--------------|----------------------|----------------------|
| Longformer   | 18%                  | 23%                  |
| MEGA         | 24%                  | 52%                  |



The results highlight the trade-offs between sequence length capacity and model efficiency. Longformer demonstrated lower performance on longer sequences, while MEGA achieved higher accuracy with shorter training times and larger batch sizes.

#### Discussion and Conclusion
- **Handling extremely long sequences:** 
  Models like MEGA required extensive GPU memory and longer training times. Longformer's efficient attention mechanisms mitigated some issues but at the cost of accuracy.
- **Fine-tuning hyperparameters for hierarchical reasoning tasks:** 
  Optimizing models required careful calibration of hyperparameters to balance precision at different depths.
- **Generalization across sequence lengths:** 
  MEGA performed better on shorter sequences and maintained robustness at 8K tokens, whereas Longformer struggled with longer contexts.

#### Conclusion
This study highlights the importance of specialized attention mechanisms for long-context hierarchical reasoning tasks. MEGA demonstrated superior performance across all tested sequence lengths, making it a promising choice for tasks involving complex hierarchical reasoning over long sequences.

### BigBird Model


The BigBird model implements a sparse attention mechanism that combines global, window, and random attention patterns to efficiently process long sequences. Training was conducted using Kaggle's T4x2 GPU environment. The model was evaluated on the ListOps dataset using three different configurations:

- **Initial Configuration:** Higher capacity model 
- **Reduced Configuration:** Higher capacity model version that uses 10 percent of training data
- **Depth-Specific Configuration:** Tested with depth-20 sequences

**Training Configurations:**
1. **Initial approach:**
   - Hidden size: 512
   - Attention heads: 8
   - Intermediate size: 2048
   - Hidden layers: 6
   - Block size: 64
   - Max position embeddings: 4096
2. **Second version:**
   - Hidden size: 8
   - Attention heads: 2-4
   - Intermediate size: 512
   - Hidden layers: 2
   - Block size: 64
   - Max position embeddings: 1024-8192

#### Training Environment
The models were trained on Kaggle with the following specifications:
- **GPU:** NVIDIA Tesla
- **Runtime:** Python 3.8
- **Libraries:** PyTorch, Transformers, sklearn

#### Results and Analysis (BigBird)

##### Performance Metrics
The BigBird model was evaluated using classification accuracy on different variations of the ListOps dataset. Performance varied across configurations:

| Configuration           | Accuracy | Training Time/Epoch | Batch Size |
|-------------------------|----------|---------------------|------------|
| Initial                 | 19.22%   | 18 minutes          | 8          |
| Second (10% test data)  | 16.43%   | 30 minutes          | 2          |
| Depth-20                | 22.19%   | 16 minutes          | 10         |

##### Training Configuration and Metrics
Different training approaches were attempted to optimize performance:

1. **Initial Configuration:**
   - Batch size: 8
   - Learning rate: 1e-4
   - Optimizer: AdamW
   - Gradient accumulation steps: 4
   - Mixed precision training
   - Best validation accuracy: 19.22%
2. **Second Configuration:**
   - Batch size: 2
   - Learning rate: 1e-4
   - Simplified architecture
   - Custom tokenizer
   - Validation accuracy: 16.43%
3. **Depth-20 Configuration:**
   - Batch size: 10
   - Learning rate: 1e-3
   - Specialized vocabulary
   - Optimizer: AdamW
   - Validation accuracy: 18.50%

##### Challenges and Limitations
1. **Computational Resources:**
   - Long training times (max was 30 minutes per epoch)
   - Memory constraints with larger batch sizes
   - Required gradient accumulation and mixed precision
2. **Model Configuration:**
   - Trade-off between model capacity and training efficiency
   - Difficulty in finding optimal hyperparameters
   - Sensitivity to sequence length and depth
3. **Performance Bottlenecks:**
   - Limited accuracy across all configurations
   - Challenges with longer sequences
   - Resource constraints affecting model scale

##### Insights and Discussion
1. **Model Scaling:**
   - Larger model configurations didn't necessarily yield better results
   - Memory-optimized versions showed comparable performance
   - Training efficiency was crucial for iteration
2. **Resource Utilization:**
   - GPU memory constraints significantly influenced design choices
   - Training time impacted experimentation capacity
   - Batch size limitations affected optimization

These findings suggest that while BigBird's sparse attention mechanism theoretically allows for processing longer sequences, practical implementation faces significant challenges in the ListOps task context.

### Comparative RNN Implementation
To establish a baseline and compare with BigBird's performance, an LSTM-based model with an attention mechanism was implemented:

**Architecture Details:**
- Base: Bidirectional LSTM
- Hidden size: 256
- Number of layers: 2
- Attention: Custom attention mechanism over LSTM outputs
- Dropout: 0.3 for regularization

**Training Configuration:**
- Batch size: 32
- Optimizer: AdamW
- Learning rate: 1e-3
- Mixed precision training: Enabled

The RNN model achieved comparable performance to BigBird, reaching a validation accuracy of 19.22%.

---

## References
- BigBird: Transformers for Longer Sequences
- Long Range Arena GitHub Repository
- Mega: Moving Average Equipped Gated Attention
- Hugging Face Models
- Reformer: The Efficient Transformer


