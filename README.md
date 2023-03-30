# License Plate Detection

- [License Plate Detection](#license-plate-detection)
  - [Introduction](#introduction)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Next tasks](#next-tasks)
  - [Resources](#resources)
  - [Contributing](#contributing)
    - [Prerequisites](#prerequisites-1)
    - [Installation](#installation-1)
      - [Clone the repository](#clone-the-repository)
      - [Install pre-commit](#install-pre-commit)
      - [Install Virtualenv](#install-virtualenv)
    - [Markdown Linting and Formatting](#markdown-linting-and-formatting)

## Introduction

TODO

## Getting started

### Prerequisites

- Python 3.9

### Installation

TODO

## Next tasks

- [ ] Add documentation to code
- [ ] Add documentation do readme
- [ ] Train with PyTorch Lightning
- [ ] Add remove cpu call
- [ ] Add early stopping

## Resources

- **Tensorflow implementation**
  https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

- **Dataset**
  https://www.kaggle.com/datasets/tbjorklund/annotated-synthetic-license-plates

- **Training with PyTorch**
  https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

- **PyTorch Reproducibility**
  https://pytorch.org/docs/stable/notes/randomness.html

- **PyTorch Optimization**
  https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

- **Swiss number plate generator**
  https://platesmania.com/ch/informer

## Contributing

### Prerequisites

- Python 3.9

### Installation

#### Clone the repository

```sh
git clone git@github.com:leonardcser/pytorch-mlops-license-plate.git
```

#### Install pre-commit

```sh
pre-commit install
```

You can learn more about `pre-commit` [here](https://pre-commit.com/).

#### Install Virtualenv

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements-dev.txt
pip install --upgrade pip
```

### Markdown Linting and Formatting

This repository uses the following VSCode:

- [`spell-right`](https://marketplace.visualstudio.com/items?itemName=ban.spellright) for spell checking markdown files.
- [`isort`](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) for sorting python imports.
