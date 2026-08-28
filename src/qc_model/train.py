"""End-to-end training, with the comparisons that make the result meaningful.

Every run reports four things side by side:

    mean predictor    what you get for free -- the floor
    ridge regression  a linear model on identical features
    neural network    the thing under test
    random split      the same network, split dishonestly

The last one is the important one. Random-split scores are what most papers
report and they are inflated, because analogues of the same scaffold land on
both sides. Printing both numbers together makes the size of that inflation
visible instead of leaving it implied.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from qc_model.data import load_target_dataset, standardize
from qc_model.metrics import evaluate, ridge_predict, ridge_regression
from qc_model.nn import MLP, fit
from qc_model.split import random_split, scaffold_split, split_report


def train_target(
    conn: sqlite3.Connection,
    target_chembl_id: str,
    symbol: str = "",
    n_bits: int = 2048,
    radius: int = 2,
    hidden: tuple[int, ...] = (256, 64),
    dropout: float = 0.6,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    patience: int = 40,
    seed: int = 0,
    also_random_split: bool = True,
    save_to: str | None = None,
    verbose: bool = True,
) -> dict:
    def say(m=""):
        if verbose:
            print(m, flush=True)

    say(f"loading {symbol or target_chembl_id}")
    data = load_target_dataset(conn, target_chembl_id, radius=radius,
                               n_bits=n_bits, verbose=verbose)
    X, y, smiles = data["X"], data["y"], data["smiles"]

    say("\nscaffold split")
    tr, va, te = scaffold_split(smiles, seed=seed)
    report = split_report(smiles, tr, va, te)
    say(f"  train {report['n_train']} / valid {report['n_valid']} / test {report['n_test']}")
    say(f"  distinct scaffolds: {report['scaffolds_train']} train, "
        f"{report['scaffolds_test']} test")
    say(f"  scaffold overlap train∩test: {report['overlap_train_test']} "
        f"({'clean' if report['overlap_train_test'] == 0 else 'LEAKING'})")

    results = {
        "target": symbol or target_chembl_id,
        "target_chembl_id": target_chembl_id,
        "n_compounds": int(len(y)),
        "n_features": int(X.shape[1]),
        "label_mean": float(y.mean()),
        "label_sd": float(y.std()),
        "split": report,
        "scaffold": {},
        "random": {},
    }

    keep: dict = {}
    results["scaffold"] = _run_split(
        X, y, tr, va, te, hidden, dropout, epochs, lr, weight_decay,
        batch_size, patience, seed, say, "SCAFFOLD SPLIT (honest)", keep=keep,
    )

    if save_to and keep:
        from qc_model.nn import save as save_model

        name = (symbol or target_chembl_id).upper()
        path = Path(save_to) / f"{name}.npz"
        save_model(
            keep["model"], path,
            train_fingerprints=X[tr][:, :n_bits],
            meta={
                "target": name,
                "target_chembl_id": target_chembl_id,
                "n_bits": n_bits, "radius": radius, "hidden": list(hidden),
                "feature_mean": keep["feature_mean"].tolist(),
                "feature_sd": keep["feature_sd"].tolist(),
                "test_r2": keep["metrics"]["r2"],
                "test_rmse": keep["metrics"]["rmse"],
                "test_spearman": keep["metrics"]["spearman"],
                "n_compounds": int(len(y)),
                "split": "scaffold",
                "n_train": int(len(tr)),
            },
        )
        say(f"\n  saved model -> {path}")

    if also_random_split:
        rtr, rva, rte = random_split(len(y), seed=seed)
        results["random"] = _run_split(
            X, y, rtr, rva, rte, hidden, dropout, epochs, lr, weight_decay,
            batch_size, patience, seed, say,
            "RANDOM SPLIT (inflated -- shown for comparison only)",
        )

    return results


def _run_split(X, y, tr, va, te, hidden, dropout, epochs, lr, weight_decay,
               batch_size, patience, seed, say, title, keep=None):
    say(f"\n{title}")
    say("-" * len(title))

    x_tr, x_va, x_te = standardize(X[tr], X[va], X[te])
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    mean_pred = np.full(len(y_te), y_tr.mean(), dtype=np.float32)
    m_mean = evaluate(mean_pred, y_te)

    w = ridge_regression(x_tr, y_tr, alpha=100.0)
    m_ridge = evaluate(ridge_predict(w, x_te), y_te)

    model = MLP(X.shape[1], hidden=hidden, dropout=dropout, seed=seed)
    say("  training network")
    history = fit(model, x_tr, y_tr, x_va, y_va, epochs=epochs, lr=lr,
                  weight_decay=weight_decay, batch_size=batch_size,
                  patience=patience, seed=seed, verbose=True, log_every=25)
    m_nn = evaluate(model.predict(x_te), y_te)

    if keep is not None:
        mean = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd[sd < 1e-8] = 1.0
        keep.update(model=model, feature_mean=mean, feature_sd=sd, metrics=m_nn)

    say("")
    say(f"  {'model':<18}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'Spearman':>10}")
    for name, m in (("mean predictor", m_mean), ("ridge", m_ridge), ("neural net", m_nn)):
        say(f"  {name:<18}{m['rmse']:>8.3f}{m['mae']:>8.3f}{m['r2']:>8.3f}{m['spearman']:>10.3f}")

    verdict = (
        "network earns its complexity" if m_nn["rmse"] < m_ridge["rmse"] - 0.01
        else "ridge is as good -- the network is not worth it here"
    )
    say(f"  -> {verdict}")

    return {
        "mean_predictor": m_mean,
        "ridge": m_ridge,
        "neural_net": m_nn,
        "best_epoch": history["best_epoch"],
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from qc_research import db

    ap = argparse.ArgumentParser(
        prog="qc_model", description="train a molecular potency model from the graph"
    )
    ap.add_argument("target", help="gene symbol (e.g. LRRK2) or ChEMBL target id")
    ap.add_argument("--db", default=str(db.DEFAULT_DB))
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", default="256,64")
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-random-split", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-to", default="data/models",
                    help="directory to write the trained model into")
    args = ap.parse_args(argv)

    conn = db.connect(args.db)

    target_id, symbol = args.target, ""
    if not args.target.upper().startswith("CHEMBL"):
        row = conn.execute(
            "SELECT symbol, chembl_id FROM target WHERE symbol = ? COLLATE NOCASE",
            (args.target,),
        ).fetchone()
        if row is None or not row["chembl_id"]:
            print(f"no ChEMBL target for {args.target!r} in the graph")
            return 1
        target_id, symbol = row["chembl_id"], row["symbol"]

    results = train_target(
        conn, target_id, symbol=symbol,
        n_bits=args.bits, radius=args.radius,
        hidden=tuple(int(h) for h in args.hidden.split(",")),
        dropout=args.dropout, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, seed=args.seed,
        save_to=args.save_to,
        also_random_split=not args.no_random_split,
    )

    if results.get("random"):
        s = results["scaffold"]["neural_net"]
        r = results["random"]["neural_net"]
        print("\n" + "=" * 60)
        print(f"  scaffold-split R2 {s['r2']:.3f}   random-split R2 {r['r2']:.3f}")
        print(f"  random split overstates R2 by {r['r2'] - s['r2']:+.3f}")
        print("  the scaffold number is the one that predicts new chemistry")
        print("=" * 60)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
