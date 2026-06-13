# CCGPFL: Correlated Chained Gaussian Processes with Focal Loss

This repository contains the implementation associated with the article:

> **Learning from crowds using a focal loss function: dealing with imbalanced annotations**

The proposed approach, named **Correlated Chained Gaussian Processes with Focal Loss (CCGPFL)**, addresses multi-class classification problems with multiple annotators. The model estimates annotator-specific reliabilities as functions of the input features while capturing dependencies among annotators through correlated latent Gaussian processes. In addition, CCGPFL incorporates a focal-loss-based variational objective to mitigate the dominance of frequent annotation patterns and improve learning when sparse annotations induce class imbalance.

---

## Abstract

> **Obtaining high-quality labeled data for supervised learning is costly, motivating the use of crowdsourcing, which distributes the annotation process across multiple workers with varying levels of expertise. A key challenge in crowdsourced data is {annotation sparsity}, as each worker labels only a limited subset of instances. This sparsity can amplify class imbalance, reduce supervision for minority classes, and bias standard cross-entropy-based models toward the majority classes.
To address this problem, we propose a correlated chained Gaussian process framework trained on a focal-loss-based variational objective ({CCGPFL}). This probabilistic framework jointly models latent ground-truth and {instance-dependent} annotator reliability while accounting for correlations among annotators. In addition, the focal-weighted objective mitigates the imbalance induced by sparse annotations by assigning greater importance to harder examples during training.
Experiments on synthetic, semi-synthetic, and fully real multi-annotator datasets show that CCGPFL achieves competitive and often superior performance relative to state-of-the-art learning-from-crowds baselines in terms of Overall Accuracy (OA) and Area Under the ROC Curve (AUC).**

---

## Repository structure

| File | Description |
| --- | --- |
| `DemoMA_MCClassification_CCGPFL.ipynb` | Synthetic demonstration of multi-class classification with multiple annotators using CCGPFL. |
| `DemoMA_MCClassification_CCGPFL_GPyTorch_SemiSynthetic.ipynb` | Semi-synthetic demonstration using the Iris dataset from the UCI Machine Learning Repository. The input features and ground-truth labels are obtained from Iris, while the annotations from multiple labelers are simulated following the methodology described in the article. |
| `ccgpfl_gpytorch.py` | Main implementation of the CCGPFL model in GPyTorch. It includes the custom likelihood, the sparse variational multi-output GP model, and the model builder. |
| `utils.py` | Auxiliary functions for data processing, annotation simulation, baseline computation, and evaluation. |

---

## Model overview

Let \(K\) denote the number of classes and \(R\) the number of annotators. CCGPFL jointly models:

1. the latent ground-truth class probabilities;
2. the instance-dependent reliability of each annotator; and
3. the dependencies among annotators through correlated latent Gaussian processes.

The implementation relies on a sparse variational multi-output Gaussian process with a Linear Model of Coregionalization (LMC) strategy. The classification component is optimized using a focal-loss-based variational objective.

The model also supports sparse annotation matrices. Missing labels can be represented using negative values and excluded from the expected log-likelihood during training.

---

## Main features

- Multi-class classification with multiple annotators.
- Instance-dependent annotator reliability estimation.
- Modeling of dependencies among annotators.
- Sparse variational Gaussian process inference.
- Linear Model of Coregionalization strategy.
- Focal-loss-based variational objective.
- Support for missing annotations.
- Synthetic and semi-synthetic demonstrations.
- Performance evaluation against annotation aggregation baselines.

---

## General requirements

The project requires:

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- PyTorch
- GPyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- TensorFlow
- tqdm
- ucimlrepo

The `ucimlrepo` package is only required for the semi-synthetic Iris demonstration. TensorFlow is currently used by some auxiliary functions included in `utils.py`.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/juliangilg/CCGPFL.git
cd CCGPFL
