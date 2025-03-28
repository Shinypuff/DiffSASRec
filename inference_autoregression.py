import argparse
import torch
import numpy as np

from model import SASRec
from utils import get_data_split


@torch.no_grad()
def autoregressive_predict(model, input_seq, itemnum, topk=10):
    model.eval()
    input_seq = input_seq.clone().detach().to(model.dev)
    batch_size, seq_len = input_seq.shape
    generated = []
    predicted_mask = torch.zeros((batch_size, itemnum + 1), dtype=torch.bool, device=model.dev)

    debug_indices = [0]

    for step in range(topk):
        log_feats = model.log2feats(input_seq)
        final_feat = log_feats[:, -1, :]
        logits = torch.matmul(final_feat, model.item_emb.weight.t())
        logits[:, 0] = -float('inf')
        logits[predicted_mask] = -float('inf')

        next_item = torch.argmax(logits, dim=-1)
        generated.append(next_item)

        for i in debug_indices:
            print(f"\n[Step {step + 1}] User {i}:")
            print(f"  Predicted item: {next_item[i].item()}")
            print(f"  Before: {input_seq[i].tolist()}")

        predicted_mask[torch.arange(batch_size), next_item] = True
        input_seq = torch.cat([input_seq[:, 1:], next_item.unsqueeze(1)], dim=1)

        for i in debug_indices:
            print(f"  After:  {input_seq[i].tolist()}")

    return torch.stack(generated, dim=1)


def evaluate_autoregressive(model, test_loader, itemnum):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq_batch, target_batch in test_loader:
            seq_batch = seq_batch.to(model.dev)
            target_batch = target_batch.to(model.dev)
            preds = autoregressive_predict(model, seq_batch, itemnum, topk=10)
            all_preds.append(preds)
            all_targets.append(target_batch)

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
    COV = len(coverage_set) / itemnum

    print("Autoregressive inference results:")
    print("HR@10: {:.4f}, NDCG@10: {:.4f}, MRR@10: {:.4f}, COV@10: {:.4f}".format(HR, NDCG, MRR, COV))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--state_dict_path", default=None, type=str)
    parser.add_argument("--users_col", default="UserId")
    parser.add_argument("--items_col", default="ProductId")
    parser.add_argument("--time_col", default="Timestamp")
    parser.add_argument("--maxlen", default=200, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--time_q", default=0.95, type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_loader, test_loader = get_data_split(args)
    train_seqs, train_targets = train_loader.dataset.tensors
    all_items = torch.cat([train_seqs.flatten(), train_targets])
    itemnum = int(torch.max(all_items).item()) + 1
    num_users = train_seqs.shape[0]

    model = SASRec(num_users, itemnum, args).to(args.device)
    model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))

    evaluate_autoregressive(model, test_loader, itemnum)
