# LARS — LLM Augmented Recommendation System

LARS is a proof of concept for a fake Twitter/X feed that uses an LLM to write posts and Thompson sampling to personalize what you see from the first like or skip.

## Contributions

- **Simulate a Twitter/X timeline.** Posts are generated, and authors and engagement counts are synthetic. A fresh profile starts with built-in prompt options and LLM-proposed styles, with no X account or imported history.
- **Learn from very few ratings.** One like or skip updates multiple preferences at once, so evidence about a topic or style carries over when it appears in a new combination.
- **Keep personalization privacy-first.** Ratings and bandit state live in local profile files, and a local model can generate posts and propose new options without sending preference data to a hosted LLM.

## Setup

From the repo root on macOS or Linux (use WSL on Windows), use [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.14:

```sh
uv python install 3.14
uv sync --locked
```

Choose one model configuration and put it in a `.env` file at the repo root. Values in `.env` override matching shell variables; the file is ignored by Git.

**Local model (recommended for privacy).** LARS needs an OpenAI-compatible Chat Completions endpoint that supports structured JSON output. For example, install [Ollama](https://ollama.com/), start it (`ollama serve` if it isn't already running), and run `ollama pull qwen2.5:7b`. Then set:

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
LARS_MODEL=qwen2.5:7b
```

`ollama` is a placeholder key for the local server. For a different server, use its base URL, key, and served model name. Leave `THENEWSAPI_TOKEN` unset for the initial run.

**Hosted model.** For OpenAI, remove any `OPENAI_BASE_URL` setting from `.env` and your shell, and set:

```dotenv
OPENAI_API_KEY=your-openai-key
LARS_MODEL=gpt-4o-mini
```

Hosted inference sends prompts to the provider. See [Privacy and data](#privacy-and-data) for what those prompts can contain.

Run either interface:

```sh
uv run lars-web    # open http://127.0.0.1:8000
uv run lars        # terminal feed
```

**Press `i` to inspect what the feed has learned.** It toggles the stats in the browser; in the terminal, type `i` and press Enter. You'll see the rating count, estimated like rates, and effective sample sizes.

## Reproduce the learning loop

1. Start with an unused profile name: `uv run lars --profile demo`. A fresh profile asks the model for initial styles before generating posts. If `demo` already exists, use a different name.
2. In the terminal, type `l` then Enter to like, or press Enter to skip. In the browser (`uv run lars-web --profile demo`), click the heart to like; the rating is recorded when the post scrolls fully off the top, and an unliked post counts as a skip.
3. Check the stats before and after the first rating. The selected options move from `0.50` to `0.67` after a like or `0.33` after a skip, with `n=1`. Prefetched posts may still reflect earlier choices.
4. Rate a few more posts, then stop and relaunch with the same profile. Press `i` to verify the ratings and estimates persisted in `data/profiles/demo.json`. You can also stop one interface and open the other with the same profile.

To repeat a run, record the repo revision, model and version, settings, and rating sequence. Exact posts and arm choices can still vary. The unit tests use mocks for external calls and need no valid API key:

```sh
OPENAI_API_KEY=test uv run python -m unittest discover -s tests
```

## How it works

- Thompson sampling picks one option per dimension (topic, style, length, and so on), and the LLM writes a post from those choices.
- Each like or skip updates the selected options' Beta-Bernoulli estimates. This shares evidence across combinations without modeling interactions. Old ratings lose weight, and the LLM can suggest new options.

LARS credits the prompt choices, while [GAMBITT (Brooks et al., 2025)](https://arxiv.org/abs/2505.16311) also models the generated output and how it relates to the reward. Learning from very few ratings is a design goal, not a benchmark result.

## Privacy and data

- **On disk:** The default profile uses `data/session.json`; `--profile NAME` uses `data/profiles/NAME.json`. Each is plain JSON containing bandit parameters, active and retired options, and a log of rated post text, feedback, and metadata. Paths are relative to the working directory. To reset, stop LARS and remove or move that profile file, or use a new name. Run only one process per profile to avoid overwriting state. Profiles share a local pool of synthetic authors.
- **Model endpoint:** Generation prompts contain the feed description and selected options. Evolution prompts can also contain option-level like rates, effective counts, retired options, and examples of liked or skipped generated posts. Use a model running on your own machine to keep those requests local.
- **Other network requests:** The browser loads synthetic-author avatars from Gravatar, and optional news context fetches headlines from The News API. To avoid both during feed use, use the terminal with a local model and leave the news token unset.
- **Web access:** The browser server binds to `127.0.0.1` by default and has no authentication. Only bind it to a LAN or Tailscale interface you trust; anyone who can reach it can view and rate posts in the selected profile. Use `LARS_PORT` to run different web profiles on different ports.

## Optional settings

- `LARS_FEED`: description of the feed the model should write for (default: `a general Twitter/X timeline`). Use a fresh profile when comparing feed descriptions.
- `LARS_TEMPERATURE`: LLM temperature (default `1.2`); set it within your model's supported range.
- `LARS_MAX_CONCURRENT`: concurrent generation and prefetch size (default `3`); try `1` for a slower local server.
- `LARS_DISCOUNT`: per-rating decay (default `0.995`); set to `1` to disable it. `LARS_EXPLORE` defaults to `1`; larger values make sampling act as though less evidence has accumulated.
- `LARS_PROPOSE_EVERY`, `LARS_MAX_LEVELS`, `LARS_RETIRE_MIN_N`, and `LARS_RETIRE_THRESHOLD` control arm evolution (defaults: `15`, `8`, `10`, and `0.99`).

For current events, set `THENEWSAPI_TOKEN` to a token from [The News API](https://www.thenewsapi.com/) and use a model endpoint that supports function calling. LARS preloads up to 10 pages on an empty cache, then refreshes with one request every 30 minutes across local processes. It keeps up to 30 stories in `data/news.json` and uses stories no older than 48 hours. The model receives headlines and descriptions only if it calls the news-cache tool. Likes and skips influence how often news-oriented options are sampled; there is no fixed news quota in the feed. Any source selected from the cache is recorded with the rating, but not shown in the feed or independently verified. With no token or fresh stories, the feed runs without sampling or crediting current-events options.
