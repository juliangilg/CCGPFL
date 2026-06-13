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


## Datasets used in the article

The experiments reported in the article cover three complementary evaluation settings:

1.**Fully synthetic datasets**, in which the input features, ground-truth labels, annotator reliabilities, and observed annotations are generated synthetically. The complete experimental setup can be reproduced by running the notebook `DemoMA_MCClassification_CCGPFL.ipynb`.
2. **Semi-synthetic datasets**, in which the input features and ground-truth labels are obtained from existing benchmark datasets, while the annotations from multiple labelers are simulated following the methodology described in the article. The datasets considered in this setting include **Iris, Occupancy, Segmentation, Skin, Tic-Tac-Toe, and Wine**, all of which are publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu). To make this repository easier to use and reproduce, the notebook `DemoMA_MCClassification_CCGPFL_GPyTorch_SemiSynthetic.ipynb` provides a complete example using the Iris dataset, including data loading, annotation simulation, model training, and performance evaluation. The notebook can also be adapted to any of the semi-synthetic datasets evaluated in the article by replacing the data-loading section and adjusting the corresponding preprocessing steps.
3. **Fully real datasets**, in which both the input features and the annotations provided by multiple labelers are obtained from real-world data sources. The experiments reported in the article include the **Music** and **Voice** datasets. The **Music** dataset is publicly available and can be accessed through the following link: `https://fprodrigues.com/software/gpc-ma-gaussian-process-classification-with-multiple-annotators/`. In contrast, the **Voice** dataset is a private resource and cannot be publicly redistributed.


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
