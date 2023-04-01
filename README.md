# License Plate Detection

- [License Plate Detection](#license-plate-detection)
  - [Introduction](#introduction)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Run the Pipeline Locally](#run-the-pipeline-locally)
  - [Run the Pipeline on a Self-Hosted Runner with CI/CD](#run-the-pipeline-on-a-self-hosted-runner-with-cicd)
    - [Manual Setup](#manual-setup)
    - [With Docker](#with-docker)
      - [Build and Run the Docker image](#build-and-run-the-docker-image)
    - [With Kubernetes](#with-kubernetes)
        - [Create a secret](#create-a-secret)
        - [Create the Kubernetes Pod](#create-the-kubernetes-pod)
  - [Next tasks](#next-tasks)
  - [Contributing](#contributing)
    - [Prerequisites](#prerequisites-1)
    - [Installation](#installation-1)
      - [Clone the repository](#clone-the-repository)
      - [Install pre-commit](#install-pre-commit)
      - [Install Virtualenv](#install-virtualenv)
    - [Markdown Linting and Formatting](#markdown-linting-and-formatting)
  - [Resources](#resources)

## Introduction

TODO

## Getting started

### Prerequisites

- Python 3.9

### Installation

TODO

## Run the Pipeline Locally

## Run the Pipeline on a Self-Hosted Runner with CI/CD

In this section, you will learn how to run the pipeline on a self-hosted runner with GitHub Actions.

You can find below three different ways of setting up the self-hosted runner with GitHub Actions.

### Manual Setup

### With Docker

#### Build and Run the Docker image

```sh
docker-compose up --build
```

### With Kubernetes

##### Create a secret

```sh
echo -n "Enter the personal access token: " &&
  read -s ACCESS_TOKEN && \
    kubectl create secret generic gh-pat-secret --from-literal=personal_access_token=$ACCESS_TOKEN
  unset ACCESS_TOKEN
```

This command does the following:

- Uses the `read` command to read the personal access token from the terminal so that it is not stored in the shell history.
- Uses the `kubectl create secret` command to create a secret named `gh-pat-secret` with the personal access token.
- Uses the `unset` command to unset the `ACCESS_TOKEN` environment variable.

> Note: Replace `<personal access token>` with your GitHub personal access token.

##### Create the Kubernetes Pod

```sh
kubectl apply -f kubernetes/cml-runner.yml
```

This command create a Kubernetes Pod named `cml-runner` with the label `cml-runner-gpu`.

## Next tasks

- [ ] Pre-generate dataset instead of generating it on the fly
- [ ] Add documentation to code
- [ ] Add documentation do readme
- [ ] Train with PyTorch Lightning
- [ ] Add remove cpu call
- [ ] Add early stopping

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
