from collections.abc import Callable


def rate(tweet: str, info: Callable[[], None]) -> int:
    """Show a tweet. Returns 1 if liked, 0 if the user scrolls on.
    Hidden command: 'i' prints experiment info without rating."""
    print(f"\n{tweet}")
    while (key := input("[l]ike / [enter] next: ").strip().lower()) == "i":
        info()
    return int(key == "l")
