#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"

UV_VERSION=0.11.30
UV_BIN="${DATA_DIR}/tools/uv-${UV_VERSION}/bin/uv"
VENV="${VENV:-${DATA_DIR}/venvs/integrad-showcase}"
export HF_HOME="${DATA_DIR}/cache/huggingface"
export TORCH_HOME="${DATA_DIR}/cache/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "setup" ]]; then
    mkdir -p \
        "${DATA_DIR}/cache/huggingface" \
        "${DATA_DIR}/cache/torch" \
        "${DATA_DIR}/cache/uv" \
        "${DATA_DIR}/python" \
        "$(dirname "${VENV}")" \
        "$(dirname "${UV_BIN}")" \
        "${DATA_DIR}/runs"
    if [[ ! -x "${UV_BIN}" ]]; then
        python -m pip install --disable-pip-version-check --no-deps \
            --prefix "${DATA_DIR}/tools/uv-${UV_VERSION}" "uv==${UV_VERSION}"
    fi
    export UV_CACHE_DIR="${DATA_DIR}/cache/uv"
    export UV_PYTHON_INSTALL_DIR="${DATA_DIR}/python"
    "${UV_BIN}" python install 3.11.15
    "${UV_BIN}" venv --clear --python 3.11.15 "${VENV}"
    "${UV_BIN}" pip install --python "${VENV}/bin/python" \
        --torch-backend cu118 -e "${PROJECT_DIR}[vision,text,graph]"
    "${VENV}/bin/python" -c \
        'import torch; assert torch.version.cuda == "11.8", torch.version.cuda; print(torch.__version__, torch.version.cuda)'
    "${VENV}/bin/python" -c \
        'from torchvision.models import ResNet50_Weights, resnet50; resnet50(weights=ResNet50_Weights.DEFAULT)'
    "${VENV}/bin/python" -c \
        'from transformers import AutoModelForSequenceClassification, AutoTokenizer; model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model)'
    "${VENV}/bin/python" -c \
        "from torch_geometric.datasets import MoleculeNet; MoleculeNet('${DATA_DIR}/datasets/bbbp', name='BBBP')"
    exit
fi

PYTHON="${VENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Environment missing at ${VENV}; submit setup.sub first." >&2
    exit 1
fi

cd "${PROJECT_DIR}"

if [[ "${REQUIRE_CUDA:-0}" == "1" ]]; then
    "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
fi
exec "${PYTHON}" -u "$@"
