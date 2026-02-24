from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

from pcgr.data import MultimodalCSVDataset, collate_fn
from pcgr.model import PCGRModel
from pcgr.graph import build_layered_graph, GraphBuildConfig
from pcgr.explain import explain_sample
from torch.utils.data import DataLoader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--no_nli", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    concept_texts = ckpt["concept_texts"]
    layers = ckpt["layers"]

    clip_name = ckpt.get("clip_name", "openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained(clip_name).to(device)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False
    processor = CLIPProcessor.from_pretrained(clip_name)

    sbert = SentenceTransformer(ckpt.get("sbert_name", "sentence-transformers/all-MiniLM-L6-v2"), device=str(device))
    sbert_emb = torch.tensor(sbert.encode(concept_texts, normalize_embeddings=True), dtype=torch.float32, device=device)

    model = PCGRModel(
        clip_model=clip,
        clip_dim=512,
        concept_texts=concept_texts,
        sbert_embed_dim=sbert_emb.shape[1],
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.concept_sbert = sbert_emb
    model.eval()

    ds = MultimodalCSVDataset(args.csv, image_root=args.image_root, clip_processor=processor)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Build graph from a pass over the dataset
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch, graph=None)
            probs.append(out.p_concept.detach().cpu())
    p_val = torch.cat(probs, dim=0)

    graph = build_layered_graph(
        concept_texts=concept_texts,
        concept_emb=model.concept_q.detach().cpu(),
        layers=layers,
        p_val=p_val,
        cfg=GraphBuildConfig(use_nli=(not args.no_nli)),
        device=device,
    )

    # Explain the first sample
    batch = next(iter(loader))
    batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch_gpu, graph=graph)
    exp = explain_sample(concept_texts, graph, out.p_agg[0].cpu(), out.attn, topk=6)

    print("y_hat (prob real):", round(exp.y_hat, 4))
    print("text:", batch["raw_text"][0])
    print("image:", batch["raw_image_path"][0])
    print("\nTop concepts:")
    for c, s in exp.top_concepts:
        print(f"- {s:.3f}  {c}")

if __name__ == "__main__":
    main()
