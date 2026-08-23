import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms.functional import to_pil_image

from integrad import adaptive_integrated_gradients, classification_score


def baseline_image(kind, image, reference, blur_radius):
    if kind == "black":
        return Image.new("RGB", image.size)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(blur_radius))
    if reference is None:
        raise ValueError("--reference-image is required for reference baseline")
    return Image.open(reference).convert("RGB")


def save_model_input(tensor, path, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    to_pil_image((tensor.cpu() * std + mean).clamp(0, 1)).save(path)


def main():
    parser = argparse.ArgumentParser(description="Export ImageNet Integrated Gradients")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", choices=("black", "blur", "reference"), default="black"
    )
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--blur-radius", type=float, default=16)
    parser.add_argument(
        "--score", action="append", choices=("logit", "probability", "margin")
    )
    parser.add_argument("--target", type=int)
    parser.add_argument("--contrast", type=int)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet50(weights=weights).eval().to(args.device)
    image = Image.open(args.image).convert("RGB")
    reference = baseline_image(
        args.baseline, image, args.reference_image, args.blur_radius
    )
    inputs = transform(image).to(args.device)
    baseline = transform(reference).to(args.device)

    with torch.no_grad():
        logits = model(inputs.unsqueeze(0))[0]
        ranking = logits.argsort(descending=True)
    target = args.target if args.target is not None else ranking[0].item()
    contrast = (
        args.contrast
        if args.contrast is not None
        else next(index.item() for index in ranking if index.item() != target)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_model_input(inputs, args.output / "input.png", transform.mean, transform.std)
    save_model_input(
        baseline, args.output / "baseline.png", transform.mean, transform.std
    )

    summaries = []
    for kind in args.score or ("logit", "probability", "margin"):
        score_fn = lambda batch, kind=kind: classification_score(
            model(batch), target, contrast, kind
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

        pixel_attributions = attributions.sum(dim=0)
        web_attributions = F.interpolate(
            pixel_attributions[None, None],
            size=(56, 56),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        payload = {
            "schema": "integrated-gradients/vision-v1",
            "image": args.image.name,
            "model": "resnet50-imagenet1k-v2",
            "baseline": args.baseline,
            "score": kind,
            "target": {"id": target, "label": weights.meta["categories"][target]},
            "contrast": {"id": contrast, "label": weights.meta["categories"][contrast]},
            "steps": steps,
            "converged": converged,
            "input_score": endpoints[1].item(),
            "baseline_score": endpoints[0].item(),
            "attribution_sum": attributions.sum().item(),
            "completeness_residual": residual.item(),
            "relative_completeness_error": abs(residual.item())
            / max(abs(score_delta.item()), args.absolute_tolerance),
            "width": 56,
            "height": 56,
            "attributions": web_attributions.cpu().flatten().tolist(),
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
        "image": args.image.name,
        "baseline": args.baseline,
        "target": weights.meta["categories"][target],
        "contrast": weights.meta["categories"][contrast],
        "results": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
