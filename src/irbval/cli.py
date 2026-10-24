"""Point d'entrée en ligne de commande : ``irbval demo`` ou ``irbval rw --pd 0.01 --lgd 0.45``."""

from __future__ import annotations

import argparse

from irbval.data import simulate_portfolio
from irbval.report import run
from irbval.rwa.irb import ExposureClass, Regime, risk_weight


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="irbval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Simule un portefeuille et produit le rapport complet")
    demo.add_argument("--out", default="reports")
    demo.add_argument("--seed", type=int, default=42)

    rw = sub.add_parser("rw", help="Pondération IRB d'une exposition")
    rw.add_argument("--pd", type=float, required=True)
    rw.add_argument("--lgd", type=float, required=True)
    rw.add_argument("--classe", default="corporate", choices=[c.value for c in ExposureClass])
    rw.add_argument("--maturite", type=float, default=2.5)
    rw.add_argument("--regime", choices=["CRR2", "CRR3"], default="CRR3")

    args = parser.parse_args(argv)
    match args.cmd:
        case "demo":
            path = run(simulate_portfolio(seed=args.seed), args.out)
            print(f"Rapport écrit : {path}")
        case "rw":
            regime = Regime.crr3() if args.regime == "CRR3" else Regime.crr2()
            value = risk_weight(args.pd, args.lgd, args.classe, maturity=args.maturite,
                                regime=regime)[0]
            print(f"RW = {value:.2%}  (RWA pour 1 M€ d'EAD : {value * 1e6:,.0f} €)")


if __name__ == "__main__":
    main()
