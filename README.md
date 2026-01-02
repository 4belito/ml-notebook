# Machine Learning Notebook

This repository contains a curated collection of Jupyter notebooks exploring
core machine learning methods and principles from a mathematically grounded perspective.

The emphasis is on theoretical justification and first-principles reasoning as a basis
for understanding, designing, and analyzing learning algorithms, rather than on
empirical trial-and-error or production-level implementation.

## Topics

- Optimization algorithms and loss landscapes
- Information-theoretic concepts (entropy, KL divergence, mutual information)
- Feature representations and selection
- Learning dynamics, inductive bias, and generalization
- Neural network building blocks and training mechanisms
- Practical PyTorch implementations derived from theoretical considerations

Most notebooks combine mathematical definitions with small, controlled experiments
to illustrate assumptions, limitations, and trade-offs underlying modern machine
learning methods.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for environment and dependency
management, and an **editable install** so that notebooks can import code from the
repository without modifying `PYTHONPATH`.

After cloning the repository, run:

```bash
uv sync
uv pip install -e .