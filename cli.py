"""CLI: expense-classifier {classify,train,report}."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import model as ml
from cascade import Classifier
from config import IGNORE_LABEL, REVIEW_LABEL, Settings, __version__
from rules import MerchantsFileError

log = logging.getLogger("expense_classifier")


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"erro: arquivo não encontrado: {p}")
    try:
        return pd.read_csv(p)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        sys.exit(f"erro: falha ao ler {p}: {exc}")


def cmd_classify(args: argparse.Namespace, settings: Settings) -> int:
    df = _read_csv(args.input)
    clf = Classifier(settings, retrain_with=None if args.no_retrain else df)
    result = clf.classify_dataframe(df)
    result.to_csv(args.output, index=False)

    n = len(result)
    review = int((result["category"] == REVIEW_LABEL).sum())
    ignored = int((result["category"] == IGNORE_LABEL).sum())
    print(f"\n{n} transações -> {args.output}")
    print(f"  classificadas : {n - review - ignored}")
    print(f"  ignoradas     : {ignored} (fatura/receita/estorno)")
    print(f"  para revisar  : {review}")
    print("\npor método:")
    print(result["method"].value_counts().to_string())

    if review:
        print(
            f"\nPara revisão (edite {settings.merchants_path} "
            f"ou {settings.labels_path}):"
        )
        pending = result.loc[
            result["category"] == REVIEW_LABEL, "description_normalized"
        ].value_counts()
        print(pending.to_string())
    return 0


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    settings.ensure_initialized()
    df = _read_csv(args.input) if args.input else None
    pipeline = ml.train(settings, df)
    return 0 if pipeline is not None else 1


def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    df = _read_csv(args.input)
    for col in ("category", "amount"):
        if col not in df.columns:
            sys.exit(f"erro: {args.input} não parece um CSV classificado (falta '{col}')")
    spend = df[~df["category"].isin([IGNORE_LABEL, REVIEW_LABEL])].copy()
    if spend.empty:
        print("nenhuma transação classificada no arquivo.")
        return 1
    spend["amount_abs"] = spend["amount"].abs()
    summary = (
        spend.groupby("category")["amount_abs"]
        .agg(total="sum", transacoes="count", ticket_medio="mean")
        .sort_values("total", ascending=False)
        .round(2)
    )
    print(summary.to_string())
    print(f"\nTotal classificado: R$ {spend['amount_abs'].sum():,.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expense-classifier",
        description="Classificador local de despesas por descrição de transação.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="diretório de merchants.json/labels.csv/model.joblib "
        "(padrão: $EXPENSE_CLASSIFIER_HOME ou ~/.config/expense-classifier)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("classify", help="classifica um CSV de transações")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="expenses_classified.csv")
    p.add_argument(
        "--no-retrain",
        action="store_true",
        help="usa o model.joblib salvo em vez de retreinar",
    )
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("train", help="(re)treina o ML a partir das regras + labels.csv")
    p.add_argument("input", nargs="?", help="CSV opcional para bootstrap adicional")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="resumo por categoria de um CSV classificado")
    p.add_argument("input")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    settings = Settings(data_dir=args.data_dir) if args.data_dir else Settings()
    try:
        return args.func(args, settings)
    except MerchantsFileError as exc:
        sys.exit(f"erro: {exc}")
    except ValueError as exc:
        sys.exit(f"erro: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
