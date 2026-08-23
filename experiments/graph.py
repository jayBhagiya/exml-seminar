import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils.smiles import x_map

from integrad import adaptive_integrated_gradients, classification_score

ATOM_FEATURE_SIZES = tuple(len(values) for values in x_map.values())


class MolecularGIN(nn.Module):
    def __init__(self, hidden_size=64, layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.atom_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_size) for size in ATOM_FEATURE_SIZES
        )
        self.convolutions = nn.ModuleList(
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )
            for _ in range(layers)
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def encode_atoms(self, atom_features):
        return sum(
            embedding(atom_features[:, column])
            for column, embedding in enumerate(self.atom_embeddings)
        )

    def forward_embeddings(self, embeddings, edge_index, batch):
        for convolution in self.convolutions:
            embeddings = F.relu(convolution(embeddings, edge_index))
            embeddings = F.dropout(embeddings, self.dropout, training=self.training)
        return self.classifier(global_mean_pool(embeddings, batch))

    def forward(self, atom_features, edge_index, batch):
        return self.forward_embeddings(
            self.encode_atoms(atom_features), edge_index, batch
        )


def split_indices(dataset, split_seed):
    indices = list(range(len(dataset)))
    labels = [int(dataset[index].y.item()) for index in indices]
    train, test = train_test_split(
        indices,
        test_size=0.2,
        random_state=split_seed,
        stratify=labels,
    )
    train, validation = train_test_split(
        train,
        test_size=0.125,
        random_state=split_seed,
        stratify=[labels[index] for index in train],
    )
    return train, validation, test


def evaluate(model, loader, device, class_weights=None):
    model.eval()
    probabilities = []
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            target = data.y.view(-1).long()
            losses.append(F.cross_entropy(logits, target, weight=class_weights).item())
            probabilities.extend(logits.softmax(dim=-1)[:, 1].cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(target.cpu().tolist())
    return {
        "loss": sum(losses) / len(losses),
        "roc_auc": roc_auc_score(labels, probabilities),
        "accuracy": sum(left == right for left, right in zip(predictions, labels))
        / len(labels),
    }


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = MoleculeNet(args.data, name="BBBP")
    train_indices, validation_indices, test_indices = split_indices(
        dataset, args.split_seed
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size
    )
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size)

    model = MolecularGIN(args.hidden_size, args.layers, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    label_counts = torch.bincount(
        torch.tensor([int(dataset[index].y.item()) for index in train_indices]),
        minlength=2,
    ).float()
    class_weights = (label_counts.sum() / (2 * label_counts)).to(args.device)

    best_auc = float("-inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(logits, data.y.view(-1).long(), weight=class_weights)
            loss.backward()
            optimizer.step()

        validation = evaluate(model, validation_loader, args.device, class_weights)
        print(json.dumps({"epoch": epoch, "validation": validation}), flush=True)
        if validation["roc_auc"] > best_auc:
            best_auc = validation["roc_auc"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    validation = evaluate(model, validation_loader, args.device, class_weights)
    test = evaluate(model, test_loader, args.device, class_weights)
    checkpoint = {
        "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "validation": validation,
        "test": test,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output / "model.pt")
    (args.output / "summary.json").write_text(
        json.dumps(
            {key: value for key, value in checkpoint.items() if key != "state_dict"},
            indent=2,
        )
        + "\n"
    )


def mean_training_embedding(model, dataset, train_indices, device):
    total = torch.zeros(model.classifier.in_features, device=device)
    count = 0
    with torch.no_grad():
        for index in train_indices:
            embeddings = model.encode_atoms(dataset[index].x.to(device))
            total += embeddings.sum(dim=0)
            count += embeddings.shape[0]
    return total / count


def repeated_graph(edge_index, graph_count, node_count, device):
    offsets = torch.arange(graph_count, device=device).view(-1, 1, 1) * node_count
    edges = edge_index.unsqueeze(0).expand(graph_count, -1, -1) + offsets
    edges = edges.permute(1, 0, 2).reshape(2, -1)
    batch = torch.arange(graph_count, device=device).repeat_interleave(node_count)
    return edges, batch


def molecule_geometry(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(molecule)
    conformer = molecule.GetConformer()
    atoms = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "id": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "x": position.x,
                "y": position.y,
            }
        )
    bonds = [
        {
            "source": bond.GetBeginAtomIdx(),
            "target": bond.GetEndAtomIdx(),
            "type": str(bond.GetBondType()),
        }
        for bond in molecule.GetBonds()
    ]
    return atoms, bonds


def export(args):
    dataset = MoleculeNet(args.data, name="BBBP")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    model = MolecularGIN(
        checkpoint["hidden_size"],
        checkpoint["layers"],
        checkpoint["dropout"],
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = dataset[args.index].to(args.device)
    with torch.no_grad():
        inputs = model.encode_atoms(data.x)
    if args.baseline == "zero":
        baseline = torch.zeros_like(inputs)
    else:
        train_indices, _, _ = split_indices(dataset, checkpoint["split_seed"])
        mean = mean_training_embedding(model, dataset, train_indices, args.device)
        baseline = mean.expand_as(inputs)

    node_count = data.num_nodes

    def graph_logits(embedding_batch):
        edges, batch = repeated_graph(
            data.edge_index,
            embedding_batch.shape[0],
            node_count,
            args.device,
        )
        return model.forward_embeddings(
            embedding_batch.reshape(-1, embedding_batch.shape[-1]),
            edges,
            batch,
        )

    with torch.no_grad():
        logits = graph_logits(inputs.unsqueeze(0))[0]
        ranking = logits.argsort(descending=True)
    target = args.target if args.target is not None else ranking[0].item()
    contrast = (
        args.contrast
        if args.contrast is not None
        else next(index.item() for index in ranking if index.item() != target)
    )
    atoms, bonds = molecule_geometry(data.smiles)
    if len(atoms) != node_count:
        raise ValueError("RDKit and PyG atom order lengths differ")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for kind in args.score or ("logit", "probability", "margin"):
        score_fn = lambda batch, kind=kind: classification_score(
            graph_logits(batch), target, contrast, kind
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
        node_attributions = attributions.sum(dim=-1).cpu()
        payload = {
            "schema": "integrated-gradients/graph-v1",
            "dataset": "BBBP",
            "index": args.index,
            "smiles": data.smiles,
            "true_label": int(data.y.item()),
            "baseline": args.baseline,
            "score": kind,
            "target": target,
            "contrast": contrast,
            "steps": steps,
            "converged": converged,
            "input_score": endpoints[1].item(),
            "baseline_score": endpoints[0].item(),
            "attribution_sum": attributions.sum().item(),
            "completeness_residual": residual.item(),
            "relative_completeness_error": abs(residual.item())
            / max(abs(score_delta.item()), args.absolute_tolerance),
            "atoms": [
                {**atom, "attribution": attribution.item()}
                for atom, attribution in zip(atoms, node_attributions, strict=True)
            ],
            "bonds": bonds,
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
        "dataset": "BBBP",
        "index": args.index,
        "smiles": data.smiles,
        "true_label": int(data.y.item()),
        "prediction": target,
        "baseline": args.baseline,
        "results": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parser():
    root = argparse.ArgumentParser(description="Train and explain a BBBP molecular GIN")
    commands = root.add_subparsers(required=True)

    train_parser = commands.add_parser("train")
    train_parser.set_defaults(run=train)
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--split-seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--patience", type=int, default=15)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--layers", type=int, default=3)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    export_parser = commands.add_parser("export")
    export_parser.set_defaults(run=export)
    export_parser.add_argument("--data", type=Path, required=True)
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--index", type=int, required=True)
    export_parser.add_argument("--baseline", choices=("zero", "mean"), default="zero")
    export_parser.add_argument(
        "--score", action="append", choices=("logit", "probability", "margin")
    )
    export_parser.add_argument("--target", type=int)
    export_parser.add_argument("--contrast", type=int)
    export_parser.add_argument("--steps", type=int, default=32)
    export_parser.add_argument("--max-steps", type=int, default=1024)
    export_parser.add_argument("--relative-tolerance", type=float, default=0.05)
    export_parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    export_parser.add_argument("--batch-size", type=int, default=16)
    export_parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.run(arguments)
