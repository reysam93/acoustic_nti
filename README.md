# Feature Selection via Graph Topology Inference for Soundscape Emotion Recognition

This repository contains the official code and data used in the paper:
**"Feature Selection via Graph Topology Inference for Soundscape Emotion Recognition"**

## Reference & Paper

- **Title**: Feature Selection via Graph Topology Inference for Soundscape Emotion Recognition
- **Authors**: Samuel Rey, Luca Martino, Roberto San Millán-Castillo, and Eduardo Morgado
- **Year**: 2025
- **Link**: [arXiv:2509.16760](https://arxiv.org/abs/2509.16760)

If you use this code or the paper's methods in your research, please consider citing it:

```bibtex
@article{rey2025feature,
  title={Feature Selection via Graph Topology Inference for Soundscape Emotion Recognition},
  author={Rey, Samuel and Martino, Luca and Millan, Roberto San and Morgado, Eduardo},
  journal={arXiv preprint arXiv:2509.16760},
  year={2025}
}
```

## Repository Structure

The main structural components of this workspace are:

```text
acoustic_nti/
├── data/                           # Directory containing datasets (EMO, ARAUS) and example error curves
├── src/                            # Python source code with core logic and optimization functions
│   ├── utils.py                    # Data loading, formatting and handling utilities
│   ├── opt.py                      # Optimization solvers and routines
│   └── update_imports.py           # Helper script to maintain backwards compatibility
├── code_matlab_original/           # Original MATLAB implementation of the G-UAED algorithms
│   ├── AUED_ext_for_non_uniform_sampling.m
│   └── Interval_AED_fun_LASSO.m
├── requirements.txt                # Python environment requirements
├── G_UAED_elbow_detector.ipynb     # Jupyter notebook implementing the G-UAED approach in Python
├── opt_lambda.ipynb                # Hyperparameter exploration and optimal lambda extraction (EMO)
├── opt_lambda_araus.ipynb          # Hyperparameter exploration and optimal lambda extraction (ARAUS)
├── sem_graph_neg.ipynb             # Structured Equation Model implementation notebook (EMO)
└── sem_graph_neg_araus.ipynb       # Structured Equation Model implementation notebook (ARAUS)
```

## Main Scripts & Notebooks

The experiments conducted in the paper can be reproduced or modified directly through the Jupyter Notebooks located in the root directory:

- `sem_graph_neg.ipynb` / `sem_graph_neg_araus.ipynb`: These are the core notebooks containing the estimation of the graphs utilizing Structural Equation Models (SEM), applied to the EMO and ARAUS datasets, respectively.
- `opt_lambda.ipynb` / `opt_lambda_araus.ipynb`: These scripts analyze and extract the optimal structural hyperparameter (`lambda`) by generating the underlying error curves for the graph optimization problem.
- `G_UAED_elbow_detector.ipynb`: A Python interactive implementation of the Generalized Universal Automatic Elbow Detection algorithm. It receives non-uniform curves, calculates geometrical optimality metrics over it to return an optimal point (the "elbow"), as well as mathematically rigorous uncertainty intervals. 

## Getting Started

To prepare the environment and be able to execute any of the Python notebooks, make sure you install the required dependencies:

```bash
pip install -r requirements.txt
```

### Execution Workflow

To execute the complete algorithm and reproduce the paper's results, follow these steps:

1. **Hyperparameter Exploration**: Run `opt_lambda.ipynb` (or `opt_lambda_araus.ipynb`) to generate the error curves and extract the potential values for the structural hyperparameter \(\lambda\).
2. **Optimal Lambda Selection**: Run `G_UAED_elbow_detector.ipynb` using the generated error curves to automatically detect the optimal \(\lambda\) value (the "elbow") and its uncertainty interval.
3. **Graph Topology Inference**: Finally, use `sem_graph_neg.ipynb` (or `sem_graph_neg_araus.ipynb`) to estimate and visualize the resulting graphs for the desired values of \(\lambda\).