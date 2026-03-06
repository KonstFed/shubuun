"""Convenience entry point at project root. Prefer: python -m itmoaudio or uv run python scripts/run.py."""

from itmoaudio import __version__


def main():
    print(f"itmoaudio {__version__}")


if __name__ == "__main__":
    main()
