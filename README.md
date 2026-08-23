# Integrated Gradients Across Modalities

Minimal PyTorch implementation of [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) for image, text, and molecular-graph classifiers.

## Features

- Batched trapezoidal path integration
- Adaptive step count with completeness checks
- Logit, probability, and class-margin targets
- ResNet-50 image, DistilBERT sentiment, and GIN molecular examples
- Compact JSON exports for visualization

## Setup

```console
$ uv venv .venv
$ uv pip install --python .venv/bin/python --torch-backend cpu -e ".[vision,text,graph]"
```

Use `--torch-backend auto` on a GPU machine.

## Test

```console
$ .venv/bin/python -m unittest discover -s tests -v
```

## Run locally

```console
$ .venv/bin/python -m experiments.vision image.jpg --output outputs/vision
$ .venv/bin/python -m experiments.text "This film is not good." --output outputs/text
$ .venv/bin/python -m experiments.graph train \
    --data data/bbbp --output outputs/graph-seed-42
$ .venv/bin/python -m experiments.graph export \
    --data data/bbbp --checkpoint outputs/graph-seed-42/model.pt \
    --index 0 --output outputs/graph-example
```

Exporters write attribution values, baseline and target metadata, integration steps, and completeness error as JSON.

## HTCondor campaign

Submit files use Docker and a shared filesystem. Set project and data locations before submission:

```console
$ export IG_PROJECT_DIR="$(pwd -P)"
$ export IG_DATA_DIR="${SCRATCH:-$HOME}/integrad-showcase"
$ mkdir -p "$IG_DATA_DIR/logs" "$IG_PROJECT_DIR/inputs/vision"
```

Add three JPG files to `inputs/vision/`. Update filenames in `condor/vision.sub` when using different images.

Create the CUDA environment and warm model caches:

```console
$ condor_submit -batch-name ig-setup condor/setup.sub
$ condor_q
```

After setup succeeds, submit independent arrays:

```console
$ condor_submit -batch-name ig-vision condor/vision.sub
$ condor_submit -batch-name ig-text condor/text.sub
$ condor_submit -batch-name ig-graph-train condor/graph_train.sub
```

After graph seed 42 succeeds:

```console
$ condor_submit -batch-name ig-graph-export condor/graph_export.sub
```

The campaign creates 6 vision jobs, 12 text jobs, 3 graph training jobs, and 16 graph export jobs. Results are stored under `$IG_DATA_DIR/runs/ig-v1`; logs are stored under `$IG_DATA_DIR/logs`.

Cluster-specific Docker mounts and GPU requirements may require small submit-file adjustments.
