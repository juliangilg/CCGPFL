# CCGPFL: Correlated Chained Gaussian Processes with Focal Loss

This repository contains the implementation associated with the article:

> **Learning from crowds using a focal loss function: dealing with imbalanced annotations**

The proposed approach, named **Correlated Chained Gaussian Processes with Focal Loss (CCGPFL)**, addresses multi-class classification problems with multiple annotators. The model estimates annotator-specific reliabilities as functions of the input features while capturing dependencies among annotators through correlated latent Gaussian processes. In addition, CCGPFL incorporates a focal-loss-based variational objective to mitigate the dominance of frequent annotation patterns and improve learning when sparse annotations induce class imbalance.

The experimental evaluation presented in the article includes fully synthetic, semi-synthetic, and fully real multi-annotator datasets. The real-world datasets used in the experiments are third-party resources and were not collected or created by the authors of this repository. Links to the original data sources are provided below whenever redistribution is not permitted.

---

## Abstract

> **Obtaining high-quality labeled data for supervised learning is costly, motivating the use of crowdsourcing, which distributes the annotation process across multiple workers with varying levels of expertise. A key challenge in crowdsourced data is annotation sparsity, as each worker labels only a limited subset of instances. This sparsity can amplify class imbalance, reduce supervision for minority classes, and bias standard cross-entropy-based models toward the majority classes.  
> To address this problem, we propose a correlated chained Gaussian process framework trained on a focal-loss-based variational objective (CCGPFL). This probabilistic framework jointly models latent ground-truth labels and instance-dependent annotator reliability while accounting for correlations among annotators. In addition, the focal-weighted objective mitigates the imbalance induced by sparse annotations by assigning greater importance to harder examples during training.  
> Experiments on synthetic, semi-synthetic, and fully real multi-annotator datasets show that CCGPFL achieves competitive and often superior performance relative to state-of-the-art learning-from-crowds baselines in terms of Overall Accuracy (OA) and Area Under the ROC Curve (AUC).**

---

## Repository structure

| File | Description |
| --- | --- |
| `DemoMA_MCClassification_CCGPFL.ipynb` | Fully synthetic demonstration of multi-class classification with multiple annotators using CCGPFL. |
| `DemoMA_MCClassification_CCGPFL_GPyTorch_SemiSynthetic.ipynb` | Semi-synthetic demonstration using the Iris dataset from the UCI Machine Learning Repository. The input features and ground-truth labels are obtained from Iris, while annotations from multiple labelers are simulated following the methodology described in the article. |
| `ccgpfl_gpytorch.py` | Main implementation of the CCGPFL model in GPyTorch. It includes the custom likelihood, the sparse variational multi-output GP model, and the model builder. |
| `utils.py` | Auxiliary functions for annotation simulation. |

---

## Model overview

Let \(K\) denote the number of classes and \(R\) the number of annotators. CCGPFL jointly models:

1. the latent ground-truth class probabilities;
2. the instance-dependent reliability of each annotator; and
3. the dependencies among annotators through correlated latent Gaussian processes.

The implementation relies on a sparse variational multi-output Gaussian process with a Linear Model of Coregionalization (LMC) strategy. The classification component is optimized using a focal-loss-based variational objective.

The model also supports sparse annotation matrices. Missing labels can be represented using negative values and excluded from the expected log-likelihood during training.

---

## Datasets used in the article

The experiments reported in the article cover three complementary evaluation settings:

1. **Fully synthetic datasets**, in which the input features, ground-truth labels, annotator reliabilities, and observed annotations are generated synthetically.
2. **Semi-synthetic datasets**, in which the input features and ground-truth labels are obtained from existing benchmark datasets, while the annotations from multiple labelers are simulated following the methodology described in the article.
3. **Fully real datasets**, in which the input features and multi-annotator labels are obtained from real-world data sources.

The benchmark and real-world datasets listed below are third-party resources. They were not collected, created, or owned by the authors of this repository. Please consult the original sources for their corresponding licenses, terms of use, and citation requirements.

### Fully synthetic datasets

| Dataset | Description | Source |
| --- | --- | --- |
| `[Insert synthetic experiment name]` | Fully synthetic multi-class dataset with simulated ground-truth labels and annotator-dependent responses. | Generated using the procedure described in the article. |
| `[Insert additional synthetic experiment name]` | `[Insert a brief description of the experimental configuration.]` | Generated using the procedure described in the article. |

### Semi-synthetic datasets

| Dataset | Description | Original source |
| --- | --- | --- |
| `Iris` | Benchmark dataset used to demonstrate the semi-synthetic workflow. The input features and ground-truth labels are real, while the annotations from multiple labelers are simulated. | `[Insert Iris dataset URL]` |
| `[Insert dataset name]` | The input features and ground-truth labels are obtained from the original dataset, while annotator responses are synthetically generated following the methodology described in the article. | `[Insert dataset URL]` |
| `[Insert dataset name]` | `[Insert a brief description.]` | `[Insert dataset URL]` |

### Fully real multi-annotator datasets

| Dataset | Description | Original source |
| --- | --- | --- |
| `[Insert dataset name]` | Real-world dataset containing input features and annotations provided by multiple labelers. | `[Insert dataset URL]` |
| `[Insert dataset name]` | `[Insert a brief description of the application domain and annotation structure.]` | `[Insert dataset URL]` |
| `[Insert dataset name]` | `[Insert a brief description.]` | `[Insert dataset URL]` |

### Data availability

The repository does not redistribute third-party datasets unless their licenses explicitly allow redistribution. To reproduce the experiments, download each dataset from its original source and follow the corresponding preprocessing instructions described in the article or in the demonstration notebooks.

---

## General requirements

The project requires:

- Python 3.9 or later
- PyTorch
- GPyTorch
- NumPy
- pandas
- scikit-learn
- Matplotlib
- ucimlrepo

The `ucimlrepo` package is only required for the semi-synthetic Iris demonstration.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/juliangilg/CCGPFL.git
cd CCGPFL
