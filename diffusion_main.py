import argparse
import os
import time

import numpy as np
import torch

from model import SASRec, SASRecWithDiffusion
from utils import (WarpSampler, build_index, data_partition, evaluate,
                   evaluate_diffusion, evaluate_diffusion_multi, evaluate_valid)


def str2bool(s):
    if s.lower() not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s.lower() == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument(
    "--model_type",
    default="vanilla",
    type=str,
    choices=["vanilla", "diffusion"],
    help="Choose 'vanilla' for standard SASRec or 'diffusion' for the diffusion-based variant",
)
parser.add_argument("--num_masks", default=10, type=int, help="Number of mask tokens to use in multi_mask evaluation")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=200, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=False, type=str2bool)
parser.add_argument("--state_dict_path", default=None, type=str)
args = parser.parse_args()

train_dir = args.dataset + "_" + args.train_dir
if not os.path.isdir(train_dir):
    os.makedirs(train_dir, exist_ok=True)

with open(os.path.join(train_dir, "args.txt"), "w") as f:
    for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
        f.write(f"{k},{v}\n")

u2i_index, i2u_index = build_index(args.dataset)

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = (len(user_train) - 1) // args.batch_size + 1

cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print("average sequence length: %.2f" % (cc / len(user_train)))

log_file = os.path.join(train_dir, "log.txt")
log_f = open(log_file, "w")
log_f.write("epoch (NDCG@10) (HR@10)\n")

sampler = WarpSampler(
    user_train,
    usernum,
    itemnum,
    batch_size=args.batch_size,
    maxlen=args.maxlen,
    n_workers=3,
)

if args.model_type == "vanilla":
    model = SASRec(usernum, itemnum, args).to(args.device)
elif args.model_type == "diffusion":
    model = SASRecWithDiffusion(usernum, itemnum, args).to(args.device)

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except Exception:
        pass

model.pos_emb.weight.data[0, :] = 0
model.item_emb.weight.data[0, :] = 0

model.train()
epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(
            torch.load(args.state_dict_path, map_location=torch.device(args.device))
        )
        tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
        epoch_start_idx = int(tail[: tail.find(".")]) + 1
    except Exception as e:
        print(
            "Failed loading state_dict. Please check file path:", args.state_dict_path
        )
        import pdb

        pdb.set_trace()

if args.inference_only:
    model.eval()
    if args.model_type == "vanilla":
        t_test = evaluate(model, dataset, args)
        print(
            "Vanilla Model Test (NDCG@10: {:.4f}, HR@10: {:.4f})".format(
                t_test[0], t_test[1]
            )
        )
    elif args.model_type == "diffusion":
        t_test = evaluate_diffusion(model, dataset, args)
        print(
            "Diffusion Model Test (NDCG@10: {:.4f}, HR@10: {:.4f})".format(
                t_test[0], t_test[1]
            )
        )
    exit(0)

if args.model_type == "vanilla":
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
elif args.model_type == "diffusion":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

best_val_ndcg, best_val_hr = 0.0, 0.0
best_test_ndcg, best_test_hr = 0.0, 0.0
T = 0.0
t0 = time.time()

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    for step in range(num_batch):
        u, seq, pos, neg = [np.array(x) for x in sampler.next_batch()]
        if args.model_type == "vanilla":
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones_like(pos_logits, device=args.device)
            neg_labels = torch.zeros_like(neg_logits, device=args.device)

            optimizer.zero_grad()

            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 and step % 10 == 0:
                print(f"Vanilla loss (epoch {epoch}, step {step}): {loss.item():.4f}")
        elif args.model_type == "diffusion":
            optimizer.zero_grad()

            seq_tensor = torch.tensor(seq, dtype=torch.long, device=args.device)
            loss = model.get_loss(seq_tensor)
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 and step % 10 == 0:
                print(f"Diffusion loss (epoch {epoch}, step {step}): {loss.item():.4f}")

    if epoch % 100 == 0:
        model.eval()
        t1 = time.time() - t0
        T += t1

        if args.model_type == "vanilla":
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print(
                f"Epoch:{epoch} Time:{T:.2f}s, Valid (NDCG@10: {t_valid[0]:.4f}, HR@10: {t_valid[1]:.4f}), Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})"
            )

            if (
                t_valid[0] > best_val_ndcg
                or t_valid[1] > best_val_hr
                or t_test[0] > best_test_ndcg
                or t_test[1] > best_test_hr
            ):
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = train_dir
                fname = "SASRec.pth"
                torch.save(model.state_dict(), os.path.join(folder, fname))
        else:
            # t_test = evaluate_diffusion(model, dataset, args)
            t_test = evaluate_diffusion_multi(model, dataset, args)
            print(
                f"Epoch:{epoch} Time:{T:.2f}s, Diffusion Test (NDCG@10: {t_test[0]:.4f}, HR@10: {t_test[1]:.4f})"
            )

            if t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                folder = train_dir
                fname = "SASRecDiffusion.pth"
                torch.save(model.state_dict(), os.path.join(folder, fname))

        log_f.write(
            str(epoch)
            + " "
            + str(round(t_test[0], 4))
            + " "
            + str(round(t_test[1], 4))
            + "\n"
        )
        log_f.flush()
        t0 = time.time()
        model.train()

    if epoch == args.num_epochs:
        folder = train_dir
        if args.model_type == "vanilla":
            fname = "SASRec.pth"
        else:
            fname = "SASRecDiffusion.pth"
        torch.save(model.state_dict(), os.path.join(folder, fname))

f.close()
sampler.close()
print("Training Done.")
