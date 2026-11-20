import os

from dotenv import load_dotenv

load_dotenv(override=True)  # before submodules read their env-based settings

for name, value in list(os.environ.items()):
    if name.startswith("LARS_"):
        os.environ.setdefault("LATER_" + name.removeprefix("LARS_"), value)
