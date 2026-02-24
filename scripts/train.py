from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image, ImageDraw

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

from pcgr.data import MultimodalCSVDataset, collate_fn
from pcgr.model import PCGRModel
from pcgr.graph import build_layered_graph, GraphBuildConfig
from pcgr.explain import explain_sample
from pcgr.utils import set_seed

def make_demo_dataset(root: Path, n: int = 200) -> Path:
    """Create a tiny synthetic dataset:
    - 'real' if text color matches image color
    - 'fake' if mismatch + exaggerated text
    """
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / "data.csv"

    colors = {
        "red": (255, 60, 60),
        "green": (60, 200, 80),
        "blue": (60, 120, 255),
        "yellow": (240, 220, 80),
    }
    color_names = list(colors.keys())

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "image_path", "label"])
        writer.writeheader()
        for i in range(n):
            real = (i % 2 == 0)
            img_color = color_names[i % len(color_names)]
            txt_color = img_color if real else color_names[(i + 1) % len(color_names)]

            # create image
            img = Image.new("RGB", (224, 224), colors[img_color])
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), img_color, fill=(0, 0, 0))
            img_name = f"{i:04d}.png"
            img_path = img_dir / img_name
            img.save(img_path)

            if real:
                text = f"Report: The image clearly shows a {txt_color} square."
                label = 1
            else:
                text = f"SHOCKING!!! The image shows a {txt_color} square!!! Totally undeniable!!!"
                label = 0

            writer.writerow({"text": text, "image_path": f"images/{img_name}", "label": label})

    return csv_path

def build_default_concepts() -> tuple[list[str], list[list[int]]]:
    """A small layered concept set (anchors -> mid -> root)."""
    L0 = [
        "Does the text use emotional language, exaggeration, or excessive punctuation?",
        "Does the text contain uncertainty or hedging words (maybe, might, could)?",
        "Is the image-text content inconsistent about key attributes (e.g., color, entity, time, place)?",
        "Is the image possibly manipulated (editing, splicing, synthetic artifacts)?",
        "Is the image out of context relative to the claim or caption?",
        "Is the source credible or does the text cite reliable sources/evidence?",
    ]
    L1 = [
        "Textual distortion is present (exaggeration, misleading framing, logical fallacy).",
        "Visual distortion is present (manipulation or non-evidential visual).",
        "Cross-modal inconsistency is present between image and text.",
        "Source reliability is low or evidence is missing.",
    ]
    L2 = [
        "Overall veracity: the multimodal claim is real (not misinformation)."
    ]
    concepts = L0 + L1 + L2
    layers = [
        list(range(0, len(L0))),                                            # L0
        list(range(len(L0), len(L0) + len(L1))),                            # L1
        [len(concepts) - 1],                                                # L2 (root)
    ]
    return concepts, layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no_nli", action="store_true", help="Disable NLI when building the concept graph.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # If no CSV provided, create demo dataset
    if not args.csv:
        demo_root = Path("data/demo")
        demo_root.mkdir(parents=True, exist_ok=True)
        csv_path = make_demo_dataset(demo_root, n=240)
        image_root = str(demo_root)
        print(f"[demo] Created dataset at {csv_path}")
    else:
        csv_path = Path(args.csv)
        image_root = args.image_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Models
    clip_name = "openai/clip-vit-base-patch32"
    clip = CLIPModel.from_pretrained(clip_name).to(device)
    clip.eval()  # freeze CLIP to keep demo light
    for p in clip.parameters():
        p.requires_grad = False
    processor = CLIPProcessor.from_pretrained(clip_name)

    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))

    concept_texts, layers = build_default_concepts()
    sbert_emb = torch.tensor(sbert.encode(concept_texts, normalize_embeddings=True), dtype=torch.float32, device=device)
    model = PCGRModel(
        clip_model=clip,
        clip_dim=512,
        concept_texts=concept_texts,
        sbert_embed_dim=sbert_emb.shape[1],
        d_model=256,
        r_rank=64,
        attn_dim=256,
        rho=0.35
    ).to(device)
    model.concept_sbert = sbert_emb
    model.init_concept_q_from_sbert()

    ds = MultimodalCSVDataset(str(csv_path), image_root=image_root, clip_processor=processor)
    n_val = max(20, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    graph = None
    cfg = GraphBuildConfig(use_nli=(not args.no_nli))

    def eval_probs() -> torch.Tensor:
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch, graph=None)  # raw p only
                probs.append(out.p_concept.detach().cpu())
        return torch.cat(probs, dim=0)  # (N, K)

    # Training
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            y = batch["label"]
            out = model(batch, graph=graph)
            # BCE on root prediction
            loss_pred = torch.nn.functional.binary_cross_entropy(out.y_hat, y)
            # orthogonality regularizer (Eq.8 spirit)
            loss_ortho = model.orthogonality_loss()
            loss = 0.9 * loss_pred + 0.1 * loss_ortho

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                acc = ((out.y_hat > 0.5).float() == y).float().mean().item()
            pbar.set_postfix(loss=float(loss.item()), acc=acc)

        # Rebuild graph after each epoch using validation predictions (build-then-infer)
        p_val = eval_probs()  # (N,K)
        graph = build_layered_graph(
            concept_texts=concept_texts,
            concept_emb=model.concept_q.detach().cpu(),  # use trainable q for semantics
            layers=layers,
            p_val=p_val,
            cfg=cfg,
            device=device,
        )
        print(f"[graph] epoch={epoch+1} built parents for {sum(len(p) for p in graph.parents)} edges; root={graph.root}")

    # Inference + explanation on one validation sample
    model.eval()
    batch = next(iter(val_loader))
    batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch_gpu, graph=graph)
    # Explain first sample in batch
    i = 0
    exp = explain_sample(concept_texts, graph, out.p_agg[i].cpu(), out.attn, topk=5)
    print("\n=== Explanation (sample 0) ===")
    print("y_hat (prob real):", round(exp.y_hat, 4))
    print("text:", batch["raw_text"][i])
    print("image:", batch["raw_image_path"][i])
    print("\nTop concepts:")
    for c, s in exp.top_concepts:
        print(f"- {s:.3f}  {c}")
    print("\nReasoning chains (child -> ... -> root):")
# Save checkpoint for reuse
ckpt_path = Path("runs/pcgr_demo.pt")
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "concept_texts": concept_texts,
    "layers": layers,
    "clip_name": clip_name,
    "sbert_name": "sentence-transformers/all-MiniLM-L6-v2",
}, ckpt_path)
print(f"\n[save] checkpoint -> {ckpt_path}")

    for chain in exp.chains:
        print("  " + "  ->  ".join([f"{s:.2f}:{c}" for c, s in chain]))

if __name__ == "__main__":
    main()
