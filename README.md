# LARS (LLM Augmented Recommendation System)

<img src="feed-preview.png" alt="Light-mode LARS For You feed on a gray background, showing six generated tweets" width="560">

**What if users could own their recommendation systems, audit why they suggest content, and write the rules those systems explore?**

LARS (LLM Augmented Recommendation System) is a proof of concept for an interpretable, fully self-hostable recommendation system designed to learn from very few ratings. The demo is a synthetic Twitter/X feed that starts with no account or imported history.

## How it works

- **Learning to pick a readable recommendation policy.** LARS continually tries multiple policies written as plain-text instructions. For each new post, Thompson sampling favors options that have earned likes while still trying options it knows less about. The LLM follows the selected instructions to write the post; likes (`1`) and skips (`0`) update the bandit without fine-tuning the LLM. LARS periodically asks the LLM for new options and replaces options that are very likely to underperform the best tested option in their dimension. You can edit the built-in options in [the source](src/lars/generation/posts.py).
- **Factored multi-arm bandit for better efficiency.** Each policy combines one option per dimension: topic, style, length, visual formatting, and mood. Think “cute cats” + “deadpan one-liner” + “under 50 characters.” The factored bandit updates each selected option from a single rating, so feedback carries across policies that share an option. The tradeoff is that the bandit doesn't model interactions between dimensions. [GAMBITT](https://arxiv.org/abs/2505.16311) also models the generated output; LARS gives credit to the prompt choices.

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

## Source map

- [Recommender](src/lars/learning/recommender.py) is the headless algorithm. `recommend()` returns text with the strategy that produced it; `learn(item, reward)` updates the bandits and evolves their options. You can supply your own generator and strategy proposer.
- [Session](src/lars/session.py) connects the recommender to profiles, background generation, and feedback IDs. Both [terminal](src/lars/main.py) and [web](src/lars/web.py) use it. The web layer adds synthetic authors and engagement counts for display.
