"""Terminal entry point and command-line options shared with the web feed."""

import argparse
from collections.abc import Callable

from later.feed.profiles import profile_path


def main() -> None:
    """Terminal feed."""
    args = parse_args("Terminal feed")
    from later.session import Session

    session = Session(profile=args.profile)
    try:
        while True:
            post = session.next()
            reward = read_rating(post["text"], lambda: print(session.info()))
            session.rate(post["id"], reward)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        session.close()
        print(f"\nsaved {session.rating_count} ratings to {session.path}")


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


def read_rating(tweet: str, show_info: Callable[[], None]) -> int:
    """Read a like or skip, letting the user inspect stats without rating."""
    print(f"\n{tweet}")
    while (key := input("[l]ike / [enter] next: ").strip().lower()) == "i":
        show_info()
    return int(key == "l")


if __name__ == "__main__":
    main()
