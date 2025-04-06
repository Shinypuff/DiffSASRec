import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SASRec, SASRecWithDiffusion
from utils import get_data_split

def str2bool(s):
    if s.lower() not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s.lower() == "true"


def save_args(args, path):
    with open(path, "w") as f:
        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            f.write(f"{k},{v}\n")


def init_model_weights(model):
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0


def evaluate(model, test_loader, args):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            seq_batch, target_batch = [x.to(args.device) for x in batch]

            if args.model_type == "vanilla":
                log_feats = model.log2feats(seq_batch)
                final_feat = log_feats[:, -1, :]
                logits = torch.matmul(final_feat, model.item_emb.weight.t())
                preds = torch.topk(logits, k=10, dim=-1).indices
            else:
                if args.diffusion_type == "multi":
                    preds = model.predict_inference_multi(
                        seq_batch.cpu().numpy(),
                        num_extra=args.num_masks,
                        max_iter=20,
                        conf_threshold=0.9
                    )
                else:
                    preds = model.predict_inference(seq_batch.cpu().numpy(), top_k=10)
                preds = torch.tensor(preds)

            all_preds.append(preds)
            all_targets.append(target_batch)

    return all_preds, all_targets


def compute_metrics(all_preds, all_targets, item_emb):
    HR = NDCG = MRR = 0.0
    total = 0
    coverage_set = set()

    for preds, targets in zip(all_preds, all_targets):
        for i in range(preds.shape[0]):
            target = targets[i].item()
            pred_list = preds[i].tolist()
            total += 1

            if target in pred_list:
                rank = pred_list.index(target)
                HR += 1
                NDCG += 1.0 / np.log2(rank + 2)
                MRR += 1.0 / (rank + 1)

            coverage_set.update(pred_list)

    HR /= total
    NDCG /= total
    MRR /= total
    COV = len(coverage_set) / item_emb.num_embeddings

    return HR, NDCG, MRR, COV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--model_type", default="vanilla", choices=["vanilla", "diffusion"])
    parser.add_argument("--num_masks", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=200, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_blocks", default=1, type=int)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--sft_num_epochs", default=10, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--state_dict_path", default=None)
    parser.add_argument("--users_col", default="UserId")
    parser.add_argument("--items_col", default="ProductId")
    parser.add_argument("--time_col", default="Timestamp")
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--time_q", default=0.95, type=float)
    parser.add_argument("--diffusion_type", default="multi", choices=["multi", "single"])
    parser.add_argument("--SFT", action="store_true")
    args = parser.parse_args()

    train_dir = args.data_path.split(".")[0] + "_" + args.train_dir
    os.makedirs(train_dir, exist_ok=True)
    save_args(args, os.path.join(train_dir, "args.txt"))

    log_file = open(os.path.join(train_dir, "log.txt"), "w")
    log_file.write("epoch HR NDCG MRR COV\n")

    train_loader, test_loader = get_data_split(args)
    train_seqs, train_targets = train_loader.dataset.tensors
    itemnum = int(torch.max(torch.cat([train_seqs.flatten(), train_targets])).item()) + 1
    num_users = train_seqs.shape[0]

    model_cls = SASRec if args.model_type == "vanilla" else SASRecWithDiffusion
    model = model_cls(num_users, itemnum, args).to(args.device)
    init_model_weights(model)

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device)),
                strict=False
            )
        except Exception as e:
            print("Failed loading state_dict. Please check file path:", args.state_dict_path)
            import pdb; pdb.set_trace()

    if args.inference_only:
        all_preds, all_targets = evaluate(model, test_loader, args)
        HR, NDCG, MRR, COV = compute_metrics(all_preds, all_targets, model.item_emb)

        print(f"Inference: Test HR@10: {HR:.4f}, NDCG@10: {NDCG:.4f}, MRR@10: {MRR:.4f}, COV@10: {COV:.4f}")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    best_test_ndcg = best_test_hr = 0.0
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for seq_batch, target_batch in train_loader:
            seq_batch, target_batch = seq_batch.to(args.device), target_batch.to(args.device)
            optimizer.zero_grad()

            if args.model_type == "vanilla":
                log_feats = model.log2feats(seq_batch)
                final_feat = log_feats[:, -1, :]
                logits = torch.matmul(final_feat, model.item_emb.weight.t())
                loss = F.cross_entropy(logits, target_batch)
            else:
                loss = model.get_loss(seq_batch)

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")

        if epoch % 5 == 0:
            all_preds, all_targets = evaluate(model, test_loader, args)
            HR, NDCG, MRR, COV = compute_metrics(all_preds, all_targets, model.item_emb)

            print(f"Epoch {epoch}: Test HR@10: {HR:.4f}, NDCG@10: {NDCG:.4f}, MRR@10: {MRR:.4f}, COV@10: {COV:.4f}")

            if NDCG > best_test_ndcg or HR > best_test_hr:
                best_test_ndcg = max(NDCG, best_test_ndcg)
                best_test_hr = max(HR, best_test_hr)
                torch.save(model.state_dict(), os.path.join(train_dir, "model.pth"))

            log_file.write(f"{epoch} {round(HR, 4)} {round(NDCG, 4)} {round(MRR, 4)} {round(COV, 4)}\n")
            log_file.flush()
            t1 = time.time() - t0
            t0 = time.time()

if __name__ == "__main__":
    main()
