from torch import Tensor


def classification_score(
    logits: Tensor,
    target: int,
    contrast: int,
    kind: str,
) -> Tensor:
    """Select one scalar classification output per batch item."""
    if kind == "logit":
        return logits[:, target]
    if kind == "probability":
        return logits.softmax(dim=-1)[:, target]
    if kind == "margin":
        return logits[:, target] - logits[:, contrast]
    raise ValueError(f"unknown score kind: {kind}")
