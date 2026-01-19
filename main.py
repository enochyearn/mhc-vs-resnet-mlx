import argparse

from src.train import run_comparison, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Surviving Depth Experiment")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["resnet", "mhc", "hc_naive"],
        default="mhc",
    )
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--compare", action="store_true", help="Run all modes")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    if args.compare:
        run_comparison(
            args.depth,
            steps=args.steps,
            width=args.width,
            batch_size=args.batch_size,
        )
    else:
        run_experiment(
            args.mode,
            args.depth,
            steps=args.steps,
            width=args.width,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
