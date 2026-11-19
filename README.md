# LARS (LLM Augmented Recommendation System)

<img src="feed-preview.png" alt="Light-mode LARS For You feed on a gray background, showing six generated tweets" width="560">

**What if users could own their recommendation systems, audit why they suggest content, and write the rules those systems explore?**

LARS (LLM Augmented Recommendation System) is a proof of concept for an interpretable, fully self-hostable recommendation system designed to learn from very few ratings. The demo is a synthetic Twitter/X feed that starts with no account or imported history.

## How it works

- **Pick a strategy, then generate.** Strategies are plain-text instructions. You can edit the built-in options in [the source](src/lars/session.py). Thompson sampling picks which instructions to try, and the LLM follows them to write a post. A like counts as `1` and a skip as `0`. The bandit learns from those ratings without fine-tuning the LLM. When there is strong evidence that an option underperforms the best tested option in its dimension, LARS retires it and asks the LLM for a replacement.
- **Learn one dimension at a time.** A factored bandit builds each strategy by picking one option per dimension: topic, style, length, visual formatting, mood, and optionally current events. Think “cute cats” + “deadpan one-liner” + “under 50 characters.” Each rating updates the selected options that are still active. This treats the dimensions independently rather than modeling interactions, so a rating for one combination informs others and the estimates are easy to inspect. [GAMBITT](https://arxiv.org/abs/2505.16311) also models the generated output; LARS credits the prompt choices.
- **Generate today, curate next.** With `THENEWSAPI_TOKEN` set, the LLM can call a news-cache tool to draw on recent headlines. The same loop could pick search instructions and use a search tool to recommend existing posts or videos. That's a possible extension.

Ratings and preferences are saved locally. Run a local model to keep inference on your machine; browser avatars still load from Gravatar.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repo root, create `.env` with `OPENAI_API_KEY=your-key` to use OpenAI. For local inference, start an OpenAI-compatible server with structured output (e.g. [Ollama](https://ollama.com/) with `ollama pull qwen2.5:7b`) and set:

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
LARS_MODEL=qwen2.5:7b
```

```sh
uv run --locked lars-web --profile demo  # http://127.0.0.1:8000
```

- **Browser:** Click the heart to like. Scrolling past records a like or skip. Press `i` to see the estimated like rates and sample sizes.
- **Terminal:** Replace `lars-web` with `lars`. Enter `l` to like, Enter alone to skip, or `i` then Enter for stats.
- **Try it:** Use a fresh profile, rate a post, and press `i` to see the update. Restart with the same profile to verify it was saved.
