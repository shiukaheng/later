from dotenv import load_dotenv

load_dotenv(override=True)  # before submodules read their env-based settings

from lars.main import main  # noqa: E402

__all__ = ["main"]
