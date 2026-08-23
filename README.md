# Integrated Gradients Across Modalities

This project implements Integrated Gradients directly with PyTorch and applies
the same attribution core to image, text, and molecular-graph classifiers. It
follows Sundararajan, Taly, and Yan's [original formulation](https://proceedings.mlr.press/v70/sundararajan17a.html).

The core accepts a differentiable scalar score, one floating-point input, and a
baseline. It returns signed attributions and the completeness residual. An
adaptive wrapper doubles path samples until the requested tolerance is met or
the configured limit is reached.

## Setup

CPU environment:

```console
$ uv venv .venv
$ uv pip install --python .venv/bin/python --torch-backend cpu -e ".[vision,text,graph]"
```

On a GPU host, replace `cpu` with `auto`. The backend is selected during
installation so the same project metadata works locally and on cluster nodes.

## Verification

```console
$ .venv/bin/python -m unittest discover -s tests -v
```

Tests cover exact linear attribution, completeness, saturation, implementation
invariance, path batching, adaptive convergence, and classification score
selection.

## Experiments

```console
$ .venv/bin/python -m experiments.vision image.jpg --output outputs/vision
$ .venv/bin/python -m experiments.text "This film is not good." --output outputs/text
$ .venv/bin/python -m experiments.graph train --data data/bbbp --output outputs/graph-seed-42
$ .venv/bin/python -m experiments.graph export --data data/bbbp \
    --checkpoint outputs/graph-seed-42/model.pt --index 0 --output outputs/graph-example
```

Each exporter writes compact JSON containing attribution values, selected
output, baseline, path-sampling count, and completeness error. Vision export
also writes display-ready input and baseline images.

## HTCondor campaign

Cluster paths follow the LSV split used by the other experiments: source code
lives under your home directory, while environments, caches, logs, datasets, and runs
live under `/data`.

Before submitting, clone this repository to
`/path/to/integrad-showcase` or sync the local checkout:

```console
$ rsync -az --exclude .git --exclude .venv \
    /path/to/integrad-showcase/ \
    <submit-node>:/path/to/integrad-showcase/
```

Create the remote directories and copy the three existing ImageNet examples
used by `condor/vision.sub`:

```console
$ ssh <submit-node> 'mkdir -p /path/to/large-storage/integrad-showcase/logs /path/to/integrad-showcase/inputs/vision'
$ rsync /path/to/images/{700a04c5c2ca6e80,1e626579f6ad7b2b,023d8b91c64faf4b}.jpg \
    <submit-node>:/path/to/integrad-showcase/inputs/vision/
```

Submit setup first. It installs the CUDA 11.8 environment and warms the model
and dataset caches:

```console
$ cd /path/to/integrad-showcase/condor
$ condor_submit -batch-name ig-setup setup.sub
$ condor_q
```

After setup completes successfully, submit three independent arrays:

```console
$ condor_submit -batch-name ig-vision vision.sub
$ condor_submit -batch-name ig-text text.sub
$ condor_submit -batch-name ig-graph-train graph_train.sub
```

`vision.sub` creates six jobs, `text.sub` creates twelve, and
`graph_train.sub` creates five independent seed runs. After seed 42 finishes,
submit sixteen graph exports:

```console
$ condor_submit -batch-name ig-graph-export graph_export.sub
$ condor_q
```

Every export evaluates logit, probability, and margin targets. Vision compares
black and blurred baselines; text compares zero and padding embeddings; graph
compares zero and training-mean embeddings. Outputs land under
`/path/to/large-storage/integrad-showcase/runs/ig-v1`.
