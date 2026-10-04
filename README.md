# Integrated Gradients Across Modalities

A minimal PyTorch implementation of [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html), applied to three kinds of classifier with the same attribution code:

- **Images:** a pretrained ResNet-50 (ImageNet)
- **Text:** DistilBERT fine-tuned for sentiment (SST-2)
- **Molecular graphs:** a small GIN trained here on BBBP (blood–brain-barrier permeability)

Written for the *Explainable Machine Learning* seminar at Saarland University. The findings are in the [project write-up](https://jaybhagiya.me/projects/integrated-gradients-across-modalities/).

## What the implementation does

- Integrates gradients along the straight path from a baseline to the input with the trapezoidal rule, in batches.
- Checks **completeness**: the attributions must sum to the score difference between input and baseline. If the residual is above tolerance, the step count doubles until it passes or reaches `--max-steps`.
- Explains three scores: the raw **logit**, the softmax **probability**, and the **margin** between the target class and a contrast class.
- Supports a choice of baseline per modality: black or blurred image, zero or `[PAD]` embeddings, zero or mean atom embeddings.

## Repository layout

| Path | Contents |
|---|---|
| `integrad/` | The Integrated Gradients core and score functions |
| `experiments/vision.py` | Image attributions for ResNet-50 |
| `experiments/text.py` | Token attributions for DistilBERT |
| `experiments/graph.py` | Trains the molecular GIN (`train`) and exports atom attributions (`export`) |
| `tests/` | Unit tests for the core, including completeness checks |
| `condor/` | HTCondor jobs for running the full campaign on a GPU cluster |

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11 or newer works.

```bash
git clone https://github.com/jayBhagiya/exml-seminar.git
cd exml-seminar
uv venv .venv

# CPU only (any machine)
uv pip install --python .venv/bin/python --torch-backend cpu -e ".[vision,text,graph]"

# or pick the CUDA build that matches your GPU driver
uv pip install --python .venv/bin/python --torch-backend auto -e ".[vision,text,graph]"
```

The extras are optional: install only `vision`, `text`, or `graph` if you need one experiment. Models and the BBBP dataset download automatically on first use (about 100 MB for ResNet-50, 270 MB for DistilBERT); no accounts or tokens are needed.

## Running the experiments

Everything below runs on a laptop CPU. Each script takes `--help`, picks the GPU automatically when one is available, and writes JSON into its `--output` folder.

### Text

```bash
.venv/bin/python -m experiments.text "This film is not good." --output outputs/text
```

About 15 seconds on a CPU. Add `--baseline pad` to integrate from `[PAD]` embeddings instead of zeros.

### Images

Use any JPG or PNG photo:

```bash
.venv/bin/python -m experiments.vision path/to/photo.jpg --output outputs/vision --baseline blur
```

About 2–3 minutes on a CPU. Baselines are `black` (default), `blur`, or `reference` with `--reference-image`.

### Molecular graphs

Train the GIN first, then explain individual molecules with the checkpoint:

```bash
.venv/bin/python -m experiments.graph train --data data/bbbp --output outputs/graph-seed-42

.venv/bin/python -m experiments.graph export --data data/bbbp \
  --checkpoint outputs/graph-seed-42/model.pt --index 802 --output outputs/graph-example
```

Training runs up to 100 epochs with early stopping and takes a few minutes on a CPU. `--index` selects a molecule from BBBP (index 802 is one of the molecules the cluster campaign explains); `--baseline mean` integrates from the mean training-set atom embedding instead of zeros.

### Outputs

Each run writes one file per explained score (`logit.json`, `probability.json`, `margin.json`) and a `summary.json`. Every score file holds the per-feature attributions, the baseline and target, the number of integration steps, and the completeness residual. Image runs also save the preprocessed `input.png` and `baseline.png`; graph training saves `model.pt` and its validation and test metrics.

## Running on an HTCondor cluster

`condor/` runs the full campaign as cluster jobs inside the `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime` Docker image: 6 image jobs, 12 text jobs, 3 graph-training seeds, and 16 graph exports. A setup job installs uv and a CUDA environment once into shared storage and pre-downloads the models and dataset; every other job reuses it.

**1. Edit the variables at the top of each `.sub` file:**

| Variable | Set it to |
|---|---|
| `project_dir` | Where this repository is cloned, on a path the worker nodes can read |
| `data_dir` | Shared storage for the environment, caches, logs, and results (plan for about 10 GB) |
| `campaign` | Optional: the folder name for this set of runs under `data_dir/runs/` |

**2. Adapt the resource lines to your cluster:**
- `requirements` selects GPUs by memory (`GPUs_GlobalMemoryMb`). Add any extra constraints your cluster needs, such as a `UidDomain` or machine pool.
- `+WantGPUHomeMounted = true` is a site-specific attribute that mounts the home directory in the container. Remove it if your cluster doesn't define it.
- Your cluster must support the Docker universe. If it doesn't, switch to `universe = vanilla` and make sure the workers have Python available for `run.sh setup`.

**3. Add input images.** `vision.sub` reads three photos from `project_dir/inputs/vision/`. The images used for the write-up aren't distributed with the repository, so add your own and update the `queue` list in `vision.sub` with their file names.

**4. Create the log folder and submit:**

```bash
mkdir -p /path/to/large-storage/integrad-showcase/logs

condor_submit -batch-name ig-setup condor/setup.sub        # once: environment, models, dataset
condor_submit -batch-name ig-vision condor/vision.sub
condor_submit -batch-name ig-text condor/text.sub
condor_submit -batch-name ig-graph-train condor/graph_train.sub
# after graph seed 42 has finished:
condor_submit -batch-name ig-graph-export condor/graph_export.sub
```

Results land in `data_dir/runs/<campaign>/` and job logs in `data_dir/logs/`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
