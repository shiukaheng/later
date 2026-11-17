from lars import store
from lars.rater import rate
from lars.session import Session


def main() -> None:
    """Terminal feed."""
    session = Session()
    try:
        while True:
            tid, tweet = session.next()
            session.rate(tid, rate(tweet, lambda: print(session.info())))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        session.close()
        print(f"\nsaved {len(session.log)} ratings to {store.PATH}")


if __name__ == "__main__":
    main()
