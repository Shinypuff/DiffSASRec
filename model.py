import numpy as np
import torch
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        x = self.conv1(inputs.transpose(-1, -2))
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x).transpose(-1, -2)
        return x + inputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(
            args.maxlen + 1, args.hidden_units, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    args.hidden_units, args.num_heads, args.dropout_rate
                )
            )
            self.forward_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            )

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        batch_size, seq_len = log_seqs.shape
        poss = (
            torch.arange(1, seq_len + 1, device=self.dev)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return self.last_layernorm(seqs)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SASRecWithDiffusion(SASRec):
    def __init__(self, user_num, item_num, args):
        super().__init__(user_num, item_num, args)
        self.item_emb = torch.nn.Embedding(
            self.item_num + 2, args.hidden_units, padding_idx=0
        )
        self.mask_token_id = item_num + 1

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.tensor(log_seqs, dtype=torch.long, device=self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = torch.tensor(
            np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1]),
            device=self.dev,
        )
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=None, is_causal=False
            )
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return self.last_layernorm(seqs)

    def forward_process(self, input_ids, eps=1e-3):
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)
        return noisy_batch, masked_indices, p_mask

    def get_loss(self, log_seqs):
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)

        noisy_batch, masked_indices, p_mask = self.forward_process(log_seqs)
        log_feats = self.log2feats(noisy_batch)
        logits = self.item_emb.weight @ log_feats.transpose(1, 2)
        logits = logits.transpose(1, 2)

        masked_logits = logits[masked_indices]
        masked_labels = log_seqs[masked_indices]

        token_loss = (
            F.cross_entropy(masked_logits, masked_labels, reduction="none")
            / p_mask[masked_indices]
        )
        return token_loss.sum() / (log_seqs.shape[0] * log_seqs.shape[1])

    def predict_inference(self, log_seqs, top_k=10, **kwargs):
        self.eval()
        with torch.no_grad():
            seq_tensor = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
            seq_tensor = seq_tensor[:, 1:]
            mask_column = torch.full(
                (seq_tensor.shape[0], 1), self.mask_token_id, dtype=torch.long, device=self.dev
            )
            seq_tensor = torch.cat([seq_tensor, mask_column], dim=1)
            log_feats = self.log2feats(seq_tensor)
            logits = torch.matmul(log_feats, self.item_emb.weight.t())
            mask_logits = logits[:, -1, :]
            _, top_k_indices = torch.topk(mask_logits, top_k, dim=-1)
            return top_k_indices.cpu().numpy()

    def predict_inference_multi(self, log_seqs, num_extra=10, max_iter=20, conf_threshold=0.9):
        self.eval()
        with torch.no_grad():
            seq_tensor = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
            seq_tensor = seq_tensor[:, num_extra:]
            mask_column = torch.full(
                (seq_tensor.size(0), num_extra), self.mask_token_id, dtype=torch.long, device=self.dev
            )
            seq_tensor = torch.cat([seq_tensor, mask_column], dim=1)
            extra_positions = list(range(seq_tensor.shape[1] - num_extra, seq_tensor.shape[1]))

            for _ in range(max_iter):
                log_feats = self.log2feats(seq_tensor)
                logits = torch.matmul(log_feats, self.item_emb.weight.t())
                extra_logits = logits[:, extra_positions, :]
                probs = torch.softmax(extra_logits, dim=-1)

                filtered = [st[st != self.mask_token_id] for st in seq_tensor]
                max_len = max(t.size(0) for t in filtered)
                padded_filtered = []
                for t in filtered:
                    if t.size(0) < max_len:
                        padding = torch.full((max_len - t.size(0),), t[-1], dtype=t.dtype, device=t.device)
                        t = torch.cat([t, padding])
                    padded_filtered.append(t)
                visible = torch.stack(padded_filtered)

                batch_size = seq_tensor.size(0)
                for i in range(batch_size):
                    tokens_in_visible = torch.unique(visible[i])
                    probs[i, :, tokens_in_visible] = 0.0

                mask = seq_tensor[:, extra_positions] == self.mask_token_id
                if mask.sum() == 0:
                    break

                confidences, predictions = torch.max(probs, dim=-1)
                update_mask = mask & (confidences >= conf_threshold)

                while update_mask.sum() == 0:
                    conf_threshold *= 0.9
                    update_mask = mask & (confidences >= conf_threshold)

                new_tokens = torch.where(update_mask, predictions, seq_tensor[:, extra_positions])
                seq_tensor[:, extra_positions] = new_tokens

            return seq_tensor.cpu().numpy()
