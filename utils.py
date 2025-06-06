import copy
import random
import sys
from collections import defaultdict
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_index(dataset_name):
    ui_mat = np.loadtxt(f"data/{dataset_name}.txt", dtype=np.int32)
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for u, i in ui_mat:
        u2i_index[u].append(i)
        i2u_index[i].append(u)

    return u2i_index, i2u_index


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, seed):
    def sample(uid):
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return uid, seq, pos, neg

    np.random.seed(seed)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = [sample(uids[counter % usernum]) for _ in range(batch_size)]
        counter += batch_size
        result_queue.put(zip(*one_batch))


class WarpSampler:
    def __init__(self, user_data, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = [
            Process(
                target=sample_function,
                args=(
                    user_data,
                    usernum,
                    itemnum,
                    batch_size,
                    maxlen,
                    self.result_queue,
                    np.random.randint(2e9),
                ),
                daemon=True
            ) for _ in range(n_workers)
        ]
        for p in self.processors:
            p.start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(fname):
    usernum, itemnum = 0, 0
    User = defaultdict(list)
    with open(f"data/{fname}.txt", "r") as f:
        for line in f:
            u, i = map(int, line.rstrip().split(" "))
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    user_train, user_valid, user_test = {}, {}, {}
    for user, items in User.items():
        if len(items) < 3:
            user_train[user], user_valid[user], user_test[user] = items, [], []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]

    return user_train, user_valid, user_test, usernum, itemnum


def evaluate(model, dataset, args):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset)
    ndcg, mrr, ht, valid_user = 0.0, 0.0, 0.0, 0.0
    users = random.sample(range(1, usernum + 1), min(usernum, 10000))

    for u in users:
        if not train[u] or not test[u]:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        for i in reversed(train[u]):
            idx -= 1
            if idx < 0:
                break
            seq[idx] = i

        rated = set(train[u]) | {0}
        item_idx = [test[u][0]] + [i for i in range(1, itemnum + 1) if i not in rated]
        predictions = -model.predict(*[np.array([u]), np.array([seq]), item_idx])[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            mrr += 1.0 / (rank + 1)
            ht += 1

        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return ndcg / valid_user, ht / valid_user, mrr / valid_user


def evaluate_valid(model, dataset, args):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset)
    ndcg, ht, mrr, valid_user = 0.0, 0.0, 0.0, 0.0
    users = random.sample(range(1, usernum + 1), min(usernum, 10000))

    for u in users:
        if not train[u] or not valid[u]:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx < 0:
                break

        rated = set(train[u]) | {0}
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array([u]), np.array([seq]), item_idx])[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            ht += 1
            mrr += 1.0 / (rank + 1)

        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return ndcg / valid_user, ht / valid_user, mrr / valid_user


def evaluate_diffusion(model, dataset, args):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset)
    ndcg, ht, mrr, valid_user = 0.0, 0.0, 0.0, 0.0
    users = (
        range(1, usernum + 1)
        if usernum <= 10000
        else np.random.choice(range(1, usernum + 1), 10000, replace=False)
    )

    for u in users:
        if not train[u] or not test[u] or not valid[u]:
            continue

        seq = np.zeros(args.maxlen, dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1
            if idx < 0:
                break

        seq[-1] = model.mask_token_id

        seq_tensor = torch.tensor(
            np.expand_dims(seq, axis=0), dtype=torch.long, device=model.dev
        )
        log_feats = model.log2feats(seq_tensor)
        logits = torch.matmul(log_feats, model.item_emb.weight.t())
        masked_logits = logits[:, -1, :].cpu().detach().numpy()[0]

        ranked_indices = np.argsort(-masked_logits)
        top10_indices = ranked_indices[:10]

        true_token = test[u][0]
        valid_user += 1

        if true_token in top10_indices:
            ht += 1
            rank = np.where(ranked_indices == true_token)[0][0]
            ndcg += 1.0 / np.log2(rank + 2)
            mrr += 1.0 / (rank + 1)

    return ndcg / valid_user, ht / valid_user, mrr / valid_user


def evaluate_diffusion_multi(model, dataset, args):
    train, valid, test, usernum, itemnum = copy.deepcopy(dataset)
    total_ndcg, total_hr, total_mrr, valid_user = 0.0, 0.0, 0.0, 0.0
    users = (
        range(1, usernum + 1)
        if usernum <= 10000
        else np.random.choice(range(1, usernum + 1), 10000, replace=False)
    )

    num_extra = args.num_masks

    from tqdm import tqdm

    for u in tqdm(users):
        if not train[u] or not valid[u] or not test[u]:
            continue

        seq = np.zeros(args.maxlen, dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for item in reversed(train[u]):
            seq[idx] = item
            idx -= 1
            if idx < 0:
                break

        seq[-num_extra:] = model.mask_token_id

        pred_seq = model.predict_inference(
            np.expand_dims(seq, axis=0),
            num_extra=num_extra,
            max_iter=20,
            conf_threshold=0.9
        )[0]

        recs = pred_seq[-num_extra:]
        true_token = test[u][0]
        valid_user += 1

        if true_token in recs:
            total_hr += 1
            rank = np.where(recs == true_token)[0][0]
            total_ndcg += 1.0 / np.log2(rank + 2)
            total_mrr += 1.0 / (rank + 1)

    if valid_user == 0:
        return 0.0, 0.0, 0.0

    return total_ndcg / valid_user, total_hr / valid_user, total_mrr / valid_user


def split_by_time(data, time_q=0.95, timeid="timestamp"):
    split_timepoint = data[timeid].quantile(q=time_q, interpolation="nearest")
    after = data.query(f"{timeid} >= @split_timepoint")
    before = data.drop(after.index)
    return before, after


def create_interaction_tensor(df: pd.DataFrame, users_col, items_col, seq_len: int):
    grouped = df.groupby([users_col])[items_col].apply(list)
    interactions = []
    for user_movies in grouped:
        user_movies = user_movies[-seq_len:]
        padded = [0] * (seq_len - len(user_movies)) + user_movies
        interactions.append(padded)
    return torch.tensor(interactions, dtype=torch.long)


def get_data_split(args):
    data = pd.read_csv(args.data_path)
    users_col, items_col, time_col = args.users_col, args.items_col, args.time_col
    test_size, sep_time, maxlen, batch_size = args.test_size, args.time_q, args.maxlen, args.batch_size

    n_users = data[users_col].nunique()
    users = data[users_col].unique()
    n_items = data[items_col].nunique()
    items = np.sort(data[items_col].unique())

    users_map = {old: new for new, old in enumerate(users)}
    items_map = {old: new for new, old in enumerate(items, start=1)}
    data[users_col] = data[users_col].map(users_map)
    data[items_col] = data[items_col].map(items_map)

    new_users = data[users_col].unique()
    test_users = np.random.choice(new_users, size=int(test_size * n_users), replace=False)

    train = data[~data[users_col].isin(test_users)]
    test = data[data[users_col].isin(test_users)]

    train_before, train_after = split_by_time(train, sep_time, time_col)
    test_before, test_after = split_by_time(test, sep_time, time_col)

    holdout_train = train_after[train_after[items_col].isin(train_before[items_col].unique())]
    holdout_train = holdout_train.sort_values(time_col).groupby(users_col, as_index=False).first()

    holdout_test = test_after[test_after[items_col].isin(test_before[items_col].unique())]
    holdout_test = holdout_test.sort_values(time_col).groupby(users_col, as_index=False).first()

    valid_train_users = np.intersect1d(train_before[users_col], holdout_train[users_col])
    valid_test_users = np.intersect1d(test_before[users_col], holdout_test[users_col])

    train_final = train_before[train_before[users_col].isin(valid_train_users)]
    train_final = train_final.sort_values([users_col, time_col])
    holdout_train = holdout_train[holdout_train[users_col].isin(valid_train_users)]

    test_final = test_before[test_before[users_col].isin(valid_test_users)]
    test_final = test_final.sort_values([users_col, time_col])
    holdout_test = holdout_test[holdout_test[users_col].isin(valid_test_users)]

    train_tensor = create_interaction_tensor(train_final, users_col, items_col, maxlen)
    holdout_train_tensor = torch.tensor(holdout_train[items_col].values)
    test_tensor = create_interaction_tensor(test_final, users_col, items_col, maxlen)
    holdout_test_tensor = torch.tensor(holdout_test[items_col].values)

    train_dataset = TensorDataset(train_tensor, holdout_train_tensor)
    test_dataset = TensorDataset(test_tensor, holdout_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
