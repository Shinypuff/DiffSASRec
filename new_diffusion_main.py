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


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument(
    "--model_type",
    default="vanilla",
    type=str,
    choices=["vanilla", "diffusion"],
    help="Choose 'vanilla' for standard SASRec or 'diffusion' for the diffusion-based variant",
)
parser.add_argument(
    "--num_masks",
    default=10,
    type=int,
    help="Number of mask tokens for diffusion inference",
)
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
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--inference_only", action="store_true")
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--users_col", default="UserId", type=str)
parser.add_argument("--items_col", default="ProductId", type=str)
parser.add_argument("--time_col", default="Timestamp", type=str)
parser.add_argument("--test_size", default=0.2, type=float)
parser.add_argument("--time_q", default=0.95, type=float)
parser.add_argument(
    "--diffusion_type", default="multi", type=str, choices=["multi", "single"]
)
parser.add_argument(
    "--SFT",
    action="store_true",
    help="Enable supervised fine-tuning after diffusion pretraining",
)
args = parser.parse_args()

train_dir = args.data_path.split(".")[0] + "_" + args.train_dir

if not os.path.isdir(train_dir):
    os.makedirs(train_dir, exist_ok=True)

with open(os.path.join(train_dir, "args.txt"), "w") as f:
    for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
        f.write(f"{k},{v}\n")

f = open(f"{train_dir}/log.txt", "w")
f.write("epoch HR NDCG MRR COV\n")

train_loader, test_loader = get_data_split(args)

train_seqs, train_targets = train_loader.dataset.tensors
all_items = torch.cat([train_seqs.flatten(), train_targets])
itemnum = int(torch.max(all_items).item()) + 1

num_users = train_seqs.shape[0]

if args.model_type == "vanilla":
    model = SASRec(num_users, itemnum, args).to(args.device)
elif args.model_type == "diffusion":
    model = SASRecWithDiffusion(num_users, itemnum, args).to(args.device)

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except Exception:
        pass

model.pos_emb.weight.data[0, :] = 0
model.item_emb.weight.data[0, :] = 0

if args.state_dict_path is not None:
    try:
        model.load_state_dict(
            torch.load(args.state_dict_path, map_location=torch.device(args.device))
        )
    except Exception as e:
        print(
            "Failed loading state_dict. Please check file path:", args.state_dict_path
        )
        import pdb

        pdb.set_trace()

if args.inference_only:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            seq_batch, target_batch = batch
            seq_batch = seq_batch.to(args.device)
            target_batch = target_batch.to(args.device)
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
                        conf_threshold=0.9,
                    )
                else:
                    preds = model.predict_inference(seq_batch.cpu().numpy(), top_k=10)
                preds = torch.tensor(preds)
            all_preds.append(preds)
            all_targets.append(target_batch)

    HR = 0.0
    NDCG = 0.0
    MRR = 0.0
    total = 0
    coverage_set = set()

    for preds, targets in zip(all_preds, all_targets):
        for i in range(preds.shape[0]):
            target = targets[i].item()
            pred_list = preds[i].tolist()
            total += 1

            if target in pred_list:
                HR += 1
                rank = pred_list.index(target)
                NDCG += 1.0 / np.log2(rank + 2)
                MRR += 1.0 / (rank + 1)

            coverage_set.update(pred_list)

    HR /= total
    NDCG /= total
    MRR /= total
    COV = len(coverage_set) / model.item_emb.num_embeddings

    print(
        "Epoch {}: Test HR@10: {:.4f}, NDCG@10: {:.4f}, MRR@10: {:.4f}, COV@10: {:.4f}".format(
            epoch, HR, NDCG, MRR, COV
        )
    )
    exit(0)

if args.model_type == "vanilla":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
elif args.model_type == "diffusion":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

best_test_ndcg, best_test_hr = 0.0, 0.0
T = 0.0
t0 = time.time()

for epoch in range(1, args.num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        seq_batch, target_batch = batch
        seq_batch = seq_batch.to(args.device)
        target_batch = target_batch.to(args.device)
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

    print(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}")

    if epoch % 5 == 0:
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                seq_batch, target_batch = batch
                seq_batch = seq_batch.to(args.device)
                target_batch = target_batch.to(args.device)

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
                            conf_threshold=0.9,
                        )
                    else:
                        preds = model.predict_inference(
                            seq_batch.cpu().numpy(), top_k=10
                        )

                    preds = torch.tensor(preds)
                all_preds.append(preds)
                all_targets.append(target_batch)

        HR = 0.0
        NDCG = 0.0
        MRR = 0.0
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
        COV = len(coverage_set) / model.item_emb.num_embeddings

        print(
            "Epoch {}: Test HR@10: {:.4f}, NDCG@10: {:.4f}, MRR@10: {:.4f}, COV@10: {:.4f}".format(
                epoch, HR, NDCG, MRR, COV
            )
        )

        if NDCG > best_test_ndcg or HR > best_test_hr:
            best_test_ndcg = max(NDCG, best_test_ndcg)
            best_test_hr = max(HR, best_test_hr)
            torch.save(model.state_dict(), os.path.join(train_dir, f"model.pth"))

        f.write(
            str(epoch)
            + " "
            + str(round(HR, 4))
            + " "
            + str(round(NDCG, 4))
            + " "
            + str(round(MRR, 4))
            + " "
            + str(round(COV, 4))
            + "\n"
        )
        f.flush()

        t1 = time.time() - t0
        T += t1
        t0 = time.time()


if args.SFT:
    print("-----Supervised Fine-Tuning-----")

    for epoch in range(1, args.sft_num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            seq_batch, target_batch = batch
            seq_batch = seq_batch.to(args.device)
            target_batch = target_batch.to(args.device)
            optimizer.zero_grad()
            if args.model_type == "vanilla":
                log_feats = model.log2feats(seq_batch)
                final_feat = log_feats[:, -1, :]
                logits = torch.matmul(final_feat, model.item_emb.weight.t())
                loss = F.cross_entropy(logits, target_batch)
            else:
                log_feats = model.log2feats(seq_batch)
                logits = torch.matmul(log_feats, model.item_emb.weight.t())
                mask_logits = logits[:, -1, :]
                loss = F.cross_entropy(mask_logits, target_batch)

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")

        if epoch % 1 == 0:
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch in test_loader:
                    seq_batch, target_batch = batch
                    seq_batch = seq_batch.to(args.device)
                    target_batch = target_batch.to(args.device)
                    if args.model_type == "vanilla":
                        log_feats = model.log2feats(seq_batch)
                        final_feat = log_feats[:, -1, :]
                        logits = torch.matmul(final_feat, model.item_emb.weight.t())
                        preds = torch.topk(logits, k=10, dim=-1).indices
                    else:
                        preds = model.predict_inference(
                            seq_batch.cpu().numpy(),
                            num_extra=args.num_masks,
                            max_iter=20,
                            conf_threshold=0.9,
                        )
                        preds = torch.tensor(preds)
                    all_preds.append(preds)
                    all_targets.append(target_batch)

            HR = 0.0
            NDCG = 0.0
            MRR = 0.0
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
            COV = len(coverage_set) / model.item_emb.num_embeddings

            print(
                "Epoch {}: Test HR@10: {:.4f}, NDCG@10: {:.4f}, MRR@10: {:.4f}, COV@10: {:.4f}".format(
                    epoch, HR, NDCG, MRR, COV
                )
            )

            f.write(
                str(epoch)
                + " "
                + str(round(HR, 4))
                + " "
                + str(round(NDCG, 4))
                + " "
                + str(round(MRR, 4))
                + " "
                + str(round(COV, 4))
                + "\n"
            )
            f.flush()

            # Save coverage_set info as JSON for Pandas inspection
            coverage_dict = {
                "epoch": epoch,
                "covered_items": sorted(list(coverage_set)),
                "num_covered": len(coverage_set),
                "coverage_ratio": round(COV, 4),
            }

            with open(
                os.path.join(train_dir, f"coverage_set_epoch={epoch}.json"), "w"
            ) as cov_file:
                json.dump(coverage_dict, cov_file, indent=2)

        if NDCG > best_test_ndcg or HR > best_test_hr:
            best_test_ndcg = max(NDCG, best_test_ndcg)
            best_test_hr = max(HR, best_test_hr)
            torch.save(
                model.state_dict(), os.path.join(train_dir, f"model_epoch={epoch}.pth")
            )

    t1 = time.time() - t0
    T += t1
    t0 = time.time()
