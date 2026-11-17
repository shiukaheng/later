from lars.arguments import parse_args
from lars.rater import rate
from lars.session import Session


def main() -> None:
    """Terminal feed."""
    args = parse_args("Terminal feed")
    session = Session(profile=args.profile)
    try:
        while True:
            post = session.next()
            a = post["author"]
            session.rate(post["id"], rate(f"{a['name']} @{a['handle']}\n{post['text']}", lambda: print(session.info())))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        session.close()
        print(f"\nsaved {len(session.log)} ratings to {session.path}")


if __name__ == "__main__":
    main()
