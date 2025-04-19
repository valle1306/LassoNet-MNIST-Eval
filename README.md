# LassoNet for Feature Selection in Neural Networks

 A comparative analysis of LassoNet for embedded feature selection in neural networks, evaluated on MNIST and Fashion-MNIST datasets.

## Project Overview
This project evaluates **LassoNet**, a neural network architecture that integrates **feature sparsity and selection** into its learning process. Unlike traditional Lasso, which is limited to linear models, LassoNet extends feature selection to **nonlinear feed-forward neural networks**. Our work focuses on reproducing and extending results from the original paper using the **MNIST** and **Fashion-MNIST** datasets.

---

##  Objective
To assess the effectiveness of LassoNet in selecting informative features and maintaining high classification accuracy, particularly in high-dimensional image datasets. We compare LassoNet against other feature selection methods such as:
- **Fisher Score**
- **Principal Feature Analysis (PFA)**
- **HSIC Lasso**

---

## Methodology

### LassoNet Formulation
LassoNet extends the L1 regularization principle by combining:
- A **residual linear path** that emphasizes feature sparsity,
- A **feedforward nonlinear network**,
- A **hierarchical constraint** enforcing that nonlinear usage of a feature is allowed only if the linear path uses it.

**Loss Function**:  
$$
\min_\theta \mathcal{L}(\theta) + \lambda \|\beta\|_1 \quad \text{subject to } \|W_j\|_\infty \leq M |\beta_j|
$$  
Where:
- \( \lambda \) is the L1 penalty,
- \( M \) is the hierarchy coefficient,
- \( \beta_j \) controls feature activity in the linear path.

### Architecture
- 1 hidden layer feedforward NN with ReLU
- Residual skip connection from input to output
- Optimized using:
  - Adam optimizer for pretraining
  - Proximal gradient descent for path following
- Early stopping with patience = 10

### Datasets
- **MNIST**: 28x28 grayscale images of handwritten digits (10 classes)
- **Fashion-MNIST**: 28x28 grayscale images of clothing items (10 classes)

### 🔍 Evaluation Workflow
1. Train LassoNet with varying regularization levels
2. Extract active features at various sparsity levels (e.g., top 50, 100, 150,...)
3. Train a downstream classifier (1-layer NN) using selected features
4. Compare classification accuracy across feature selection methods

---

## 📊 Results & Findings

| Features (k) | LassoNet Accuracy | HSIC Lasso        | Fisher Score | PFA  |
|--------------|------------------:|------------------:|--------------:|------:|
| 50           | 🥈 High           | 🥇 Slightly higher | Medium        | Low  |
| 100          | 🥇 Best           | Slightly lower     | Lower         | Low  |
| 200+         | 🥇 Best           | Comparable         | Lower         | Lower|

- LassoNet performs **consistently well** across various levels of feature sparsity.
- Compared to HSIC-Lasso, LassoNet is **competitive or superior** with significantly **less training overhead**.
- Feature selection is **embedded**, not post hoc—leading to better integration in training loops.

---

##  Advantages of LassoNet
- Embedded feature selection with **no need for post-processing**
- Works on **nonlinear models** unlike traditional Lasso
- Achieves high **accuracy even with few features**
- Improves **interpretability** and reduces overfitting

## Limitations
- Slower when selecting a small fixed number of features
- Lacks statistical inference (e.g., p-values) for feature significance

---

## 📚 Reference

Lemhadri, I., Ruan, F., & Tibshirani, R. J. (2021). [LassoNet: A neural network with feature sparsity](https://arxiv.org/abs/2009.13605). *ICLR 2021*.
