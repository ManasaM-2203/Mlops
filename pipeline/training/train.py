import argparse
import json
import pickle

import mlflow
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, precision_score, recall_score

from pipeline.config import ROOT, load_params
from pipeline.training.dataset import NERDataset, BertNERDataset


def load_splits(processed_path):
    base = ROOT / processed_path
    with open(base / "train.pkl", "rb") as f:
        train = pickle.load(f)
    with open(base / "val.pkl", "rb") as f:
        val = pickle.load(f)
    with open(base / "test.pkl", "rb") as f:
        test = pickle.load(f)
    with open(base / "word2idx.json") as f:
        word2idx = json.load(f)
    with open(base / "tag2idx.json") as f:
        tag2idx = json.load(f)
    with open(base / "idx2tag.json") as f:
        idx2tag = json.load(f)
    return train, val, test, word2idx, tag2idx, idx2tag


def compute_metrics(true_tags, pred_tags):
    return {
        "f1": f1_score(true_tags, pred_tags),
        "precision": precision_score(true_tags, pred_tags),
        "recall": recall_score(true_tags, pred_tags),
    }


def tags_to_labels(tag_ids, lengths, idx2tag):
    results = []
    for seq, length in zip(tag_ids, lengths):
        results.append([idx2tag.get(str(t), "O") for t in seq[:length]])
    return results


def train_crf(params, train_sents, val_sents, test_sents):
    from pipeline.models.crf_model import CRFModel, sent_labels

    crf_params = params["crf"]
    model = CRFModel(**crf_params)
    model_dir = ROOT / "saved_models" / "crf"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.crfsuite"

    print("[train:crf] training")
    model.train(train_sents, model_path)
    model.load(model_path)

    pred = model.predict(test_sents)
    true = [sent_labels(s) for s in test_sents]
    metrics = compute_metrics(true, pred)

    print(f"[train:crf] F1={metrics['f1']:.4f}")
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_bilstm_crf(params, train_sents, val_sents, test_sents, word2idx, tag2idx, idx2tag):
    from pipeline.models.bilstm_crf import BiLSTMCRF

    hp = params["bilstm_crf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train:bilstm_crf] using device: {device}")

    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        tagset_size=len(tag2idx),
        embedding_dim=hp["embedding_dim"],
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
    train_ds = NERDataset(train_sents, word2idx, tag2idx)
    val_ds = NERDataset(val_sents, word2idx, tag2idx)
    test_ds = NERDataset(test_sents, word2idx, tag2idx)
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=hp["batch_size"])

    model_dir = ROOT / "saved_models" / "bilstm_crf"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    for epoch in range(hp["epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            loss = model.loss(ids, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = _evaluate_bilstm(model, val_loader, idx2tag, device)
        print(f"[train:bilstm_crf] epoch {epoch+1}/{hp['epochs']} loss={total_loss/len(train_loader):.4f} val_f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), model_dir / "model.pt")

    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    test_metrics = _evaluate_bilstm(model, test_loader, idx2tag, device)
    print(f"[train:bilstm_crf] test F1={test_metrics['f1']:.4f}")

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def _evaluate_bilstm(model, loader, idx2tag, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            tags = batch["tags"]
            mask = batch["mask"].to(device)
            lengths = batch["length"].tolist()

            preds = model.predict(ids, mask).cpu().tolist()
            trues = tags.tolist()

            all_true.extend(tags_to_labels(trues, lengths, idx2tag))
            all_pred.extend(tags_to_labels(preds, lengths, idx2tag))

    return compute_metrics(all_true, all_pred)


def train_distilbert_ner(params, train_sents, val_sents, test_sents, tag2idx, idx2tag):
    from pipeline.models.distilbert_ner import DistilBertNER, DistilBertNERTokenizer
    return _train_transformer_ner(
        name="distilbert_ner",
        hp=params["distilbert_ner"],
        model_cls=DistilBertNER,
        tokenizer_cls=DistilBertNERTokenizer,
        train_sents=train_sents,
        val_sents=val_sents,
        test_sents=test_sents,
        tag2idx=tag2idx,
        idx2tag=idx2tag,
    )


def train_bert_ner(params, train_sents, val_sents, test_sents, tag2idx, idx2tag):
    from pipeline.models.bert_ner import BertNER, BertNERTokenizer
    return _train_transformer_ner(
        name="bert_ner",
        hp=params["bert_ner"],
        model_cls=BertNER,
        tokenizer_cls=BertNERTokenizer,
        train_sents=train_sents,
        val_sents=val_sents,
        test_sents=test_sents,
        tag2idx=tag2idx,
        idx2tag=idx2tag,
    )


def _train_transformer_ner(name, hp, model_cls, tokenizer_cls, train_sents, val_sents, test_sents, tag2idx, idx2tag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train:{name}] using device: {device}")

    tokenizer = tokenizer_cls(hp["model_name"], hp["max_len"])
    model = model_cls(hp["model_name"], len(tag2idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_ds = BertNERDataset(train_sents, tokenizer, tag2idx)
    val_ds = BertNERDataset(val_sents, tokenizer, tag2idx)
    test_ds = BertNERDataset(test_sents, tokenizer, tag2idx)
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=hp["batch_size"])

    model_dir = ROOT / "saved_models" / name
    model_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    for epoch in range(hp["epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.view(-1, len(tag2idx)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = _evaluate_bert(model, val_loader, idx2tag, tag2idx, device)
        print(f"[train:{name}] epoch {epoch+1}/{hp['epochs']} loss={total_loss/len(train_loader):.4f} val_f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), model_dir / "model.pt")
            tokenizer.tokenizer.save_pretrained(str(model_dir / "tokenizer"))

    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    test_metrics = _evaluate_bert(model, test_loader, idx2tag, tag2idx, device)
    print(f"[train:{name}] test F1={test_metrics['f1']:.4f}")

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def _evaluate_bert(model, loader, idx2tag, tag2idx, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().tolist()
            trues = labels.tolist()

            for pred_seq, true_seq in zip(preds, trues):
                p, t = [], []
                for pi, ti in zip(pred_seq, true_seq):
                    if ti == -100:
                        continue
                    p.append(idx2tag.get(str(pi), "O"))
                    t.append(idx2tag.get(str(ti), "O"))
                all_pred.append(p)
                all_true.append(t)

    return compute_metrics(all_true, all_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["crf", "bilstm_crf", "bert_ner", "distilbert_ner"])
    args = parser.parse_args()

    params = load_params()
    train_sents, val_sents, test_sents, word2idx, tag2idx, idx2tag = load_splits(
        params["data"]["processed_path"]
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=args.model):
        if args.model == "crf":
            mlflow.log_params(params["crf"])
            metrics = train_crf(params, train_sents, val_sents, test_sents)
        elif args.model == "bilstm_crf":
            mlflow.log_params(params["bilstm_crf"])
            metrics = train_bilstm_crf(
                params, train_sents, val_sents, test_sents, word2idx, tag2idx, idx2tag
            )
        elif args.model == "bert_ner":
            mlflow.log_params(params["bert_ner"])
            metrics = train_bert_ner(
                params, train_sents, val_sents, test_sents, tag2idx, idx2tag
            )
        elif args.model == "distilbert_ner":
            mlflow.log_params(params["distilbert_ner"])
            metrics = train_distilbert_ner(
                params, train_sents, val_sents, test_sents, tag2idx, idx2tag
            )

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(ROOT / "saved_models" / args.model))


if __name__ == "__main__":
    main()
