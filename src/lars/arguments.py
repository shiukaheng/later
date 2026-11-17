import argparse

from lars.store import profile_path


def parse_args(description: str, *, web: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)

    def profile_name(value: str) -> str:
        try:
            profile_path(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        return value

    parser.add_argument("--profile", default="default", type=profile_name, help="separate feed profile (default: default)")
    if web:
        parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    return parser.parse_args()
