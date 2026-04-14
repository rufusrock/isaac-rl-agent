from __future__ import annotations

import argparse

from binding_rl_agent.diagnostics import save_diagnostic_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save diagnostic images from the Isaac live capture pipeline."
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional substring to match the game window title.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where diagnostic PNGs will be written.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=84,
        help="Width of the resized RL observation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=84,
        help="Height of the resized RL observation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = save_diagnostic_images(
        output_dir=args.output_dir,
        title_substring=args.title,
        width=args.width,
        height=args.height,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
