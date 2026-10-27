# Regression Analysis, Resampling Methods, and Gradient Descent

This repository contains the implementation, numerical analysis, and scientific report for **Project 1** of **FYS-STK3155 / FYS4155: Data Analysis and Machine Learning** at the University of Oslo (UiO).

The project explores linear regression techniques, resampling methods for model evaluation and selection, and first-order optimization routines applied to **Runge's function**:

$$f(x) = \frac{1}{1 + 25x^2}, \quad x \in [-1, 1]$$

with added Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

---

## Key Features & Methodologies

### 1. Regression Models
* **Ordinary Least Squares (OLS):** Closed-form solution via matrix pseudoinverse and SVD; polynomial fitting up to order 15+.
* **Ridge Regression:** Analytical solution with $l_2$ regularization parameter $\lambda$; study of singular-value shrinkage modes.
* **Lasso Regression:** Custom subgradient descent and Scikit-Learn comparisons for $l_1$-penalized regression without a closed-form solution.

### 2. Resampling & Model Evaluation
* **Bias-Variance Tradeoff:** Theoretical derivation and numerical investigation of the bias-variance decomposition using non-parametric **Bootstrap** resampling.
* **Cross-Validation:** $k$-Fold cross-validation ($k=5, 10$) implemented with proper in-loop feature scaling to prevent data leakage.
* **Metrics:** Evaluated via Mean Squared Error (MSE) and $R^2$ score.

### 3. Optimization Techniques
* **Gradient Descent (GD) & Stochastic Gradient Descent (SGD):** Implementation of custom optimizers with fixed and adaptive learning rates.
* **Gradient Computation:** Exact analytical gradients vs. Automatic Differentiation via `JAX` (`jax.grad`) and `Autograd`.
* **Advanced Optimizers:** Comparative evaluation of Momentum, AdaGrad, RMSprop, and Adam optimizers across convergence speeds and learning-rate stability relative to the Hessian matrix $\frac{2}{n}X^T X$.

---

## Repository Structure

```text
.
├── Results/
│   ├── Code/
│   │   ├── basic_code/       # OLS, Ridge, Resampling & basic GD implementations
│   │   ├── task_g/           # Lasso regression solvers and subgradient methods
│   │   ├── task_h/           # Stochastic Gradient Descent (SGD) & learning rate schedules
│   │   └── task_i/           # Final comparative cross-validation model selection
│   ├── Figs/                 # Plots and figures generated for the analysis
│   ├── Result_overview/      # Compiled outputs and summary evaluations
│   └── Project_1_Machine_Learning.pdf # Final scientific report (PDF)
├── Structure/                # Workflow schematics and organizational files
├── Tasks/
│   ├── Project1.ipynb        
│   ├── Project1.pdf          # Official project problem description
│   └── Project1.tex          
└── README.md                 # Project documentation



The Collaborators for Project 1 were: Steffen Pablo Krieg, Johannes Schäfers, Benedikt Schneider, Ferdinand Nilius.

A quick overview of our results (mostly code output with numerical results) is located in Results_overview/Results.txt
