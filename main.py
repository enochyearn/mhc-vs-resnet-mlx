import argparse

from src.train import plot_mhc_matrix, plot_results, run_comparison, run_experiment, save_metrics


def main():
    parser = argparse.ArgumentParser(description="Surviving Depth Experiment")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["resnet", "hc_causal", "mhc_causal", "hc", "mhc"],
        default="mhc_causal",
    )
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--compare", action="store_true", help="Run all modes")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--streams", type=int, default=1, help="Number of streams")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="Weight decay for AdamW"
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="LR warmup steps")
    parser.add_argument("--no-compile", action="store_true", help="Disable mx.compile")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--no-schedule", action="store_true", help="Disable LR schedule")

    args = parser.parse_args()

    if args.compare:
        run_comparison(
            args.depth,
            steps=args.steps,
            width=args.width,
            batch_size=args.batch_size,
            seed=args.seed,
            streams=args.streams,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            compile_step=not args.no_compile,
            dropout=args.dropout,
            use_schedule=not args.no_schedule,
        )
    else:
        history, model = run_experiment(
            args.mode,
            args.depth,
            steps=args.steps,
            width=args.width,
            batch_size=args.batch_size,
            seed=args.seed,
            streams=args.streams,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            compile_step=not args.no_compile,
            dropout=args.dropout,
            use_schedule=not args.no_schedule,
        )
        config = {
            "depth": args.depth,
            "width": args.width,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "streams": args.streams,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "compile_step": not args.no_compile,
            "dropout": args.dropout,
            "use_schedule": not args.no_schedule,
            "modes": [args.mode],
        }
        histories = {args.mode: history}
        save_metrics(histories, config)
        plot_results(histories)
        if args.mode in ["mhc", "mhc_causal"]:
            plot_mhc_matrix(model)


if __name__ == "__main__":
    main()
