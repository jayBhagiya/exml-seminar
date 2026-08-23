import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from integrad import adaptive_integrated_gradients, classification_score


def main():
    parser = argparse.ArgumentParser(description="Export text Integrated Gradients")
    parser.add_argument("text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    parser.add_argument("--baseline", choices=("zero", "pad"), default="zero")
    parser.add_argument(
        "--score", action="append", choices=("logit", "probability", "margin")
    )
    parser.add_argument("--target", type=int)
    parser.add_argument("--contrast", type=int)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForSequenceClassification.from_pretrained(args.model)
        .eval()
        .to(args.device)
    )
    encoded = tokenizer(
        args.text,
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=True,
        max_length=args.max_length,
    )
    special_mask = encoded.pop("special_tokens_mask")[0].bool().to(args.device)
    encoded = {name: value.to(args.device) for name, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        ranking = logits.argsort(descending=True)
        inputs = model.get_input_embeddings()(encoded["input_ids"])[0]
    target = args.target if args.target is not None else ranking[0].item()
    contrast = (
        args.contrast
        if args.contrast is not None
        else next(index.item() for index in ranking if index.item() != target)
    )

    baseline = inputs.clone()
    if args.baseline == "zero":
        baseline[~special_mask] = 0
    else:
        pad_id = tokenizer.pad_token_id
        with torch.no_grad():
            pad = model.get_input_embeddings()(
                torch.tensor([pad_id], device=args.device)
            )[0]
        baseline[~special_mask] = pad

    attention_mask = encoded["attention_mask"]

    def model_logits(batch):
        mask = attention_mask.expand(batch.shape[0], -1)
        return model(inputs_embeds=batch, attention_mask=mask).logits

    args.output.mkdir(parents=True, exist_ok=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    summaries = []
    for kind in args.score or ("logit", "probability", "margin"):
        score_fn = lambda batch, kind=kind: classification_score(
            model_logits(batch), target, contrast, kind
        )
        attributions, residual, steps, converged = adaptive_integrated_gradients(
            score_fn,
            inputs,
            baseline,
            steps=args.steps,
            max_steps=args.max_steps,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
            batch_size=args.batch_size,
        )
        with torch.no_grad():
            endpoints = score_fn(torch.stack((baseline, inputs))).cpu()
        score_delta = endpoints[1] - endpoints[0]
        token_attributions = attributions.sum(dim=-1).cpu()

        payload = {
            "schema": "integrated-gradients/text-v1",
            "model": args.model,
            "baseline": args.baseline,
            "score": kind,
            "target": {"id": target, "label": model.config.id2label[target]},
            "contrast": {"id": contrast, "label": model.config.id2label[contrast]},
            "steps": steps,
            "converged": converged,
            "input_score": endpoints[1].item(),
            "baseline_score": endpoints[0].item(),
            "attribution_sum": attributions.sum().item(),
            "completeness_residual": residual.item(),
            "relative_completeness_error": abs(residual.item())
            / max(abs(score_delta.item()), args.absolute_tolerance),
            "tokens": [
                {
                    "text": token,
                    "attribution": attribution.item(),
                    "special": bool(is_special),
                }
                for token, attribution, is_special in zip(
                    tokens, token_attributions, special_mask.cpu(), strict=True
                )
            ],
        }
        (args.output / f"{kind}.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        summaries.append(
            {
                key: payload[key]
                for key in (
                    "score",
                    "steps",
                    "converged",
                    "input_score",
                    "baseline_score",
                    "attribution_sum",
                    "completeness_residual",
                    "relative_completeness_error",
                )
            }
        )

    summary = {
        "text": args.text,
        "baseline": args.baseline,
        "target": model.config.id2label[target],
        "contrast": model.config.id2label[contrast],
        "results": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
