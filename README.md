# LARS (LLM Augmented Recommendation System)

<img src="feed-preview.png" alt="Light-mode LARS For You feed on a gray background, showing six generated tweets" width="560">

**What if users could own their recommendation systems, audit why they suggest content, and write the rules those systems explore?**

LARS (LLM Augmented Recommendation System) is a proof of concept for an interpretable, fully self-hostable recommendation system designed to learn from very few ratings. The demo is a synthetic Twitter/X feed that starts with no account or imported history.

## How it works

- **Multi-armed bandit on readable suggestion policies.** Thompson sampling explores plain-text instructions, and the LLM turns them into posts. Likes and skips update the bandit without fine-tuning the LLM. The LLM proposes new options as the bandit retires weak ones.
- **Factored multi-arm bandit for better efficiency.** Each post combines a topic, style, visual format, and mood. One rating updates each selected option, sharing evidence across combinations without modeling interactions. [GAMBITT](https://arxiv.org/abs/2505.16311) also models the generated output.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repo root, create `.env` with `OPENAI_API_KEY=your-key` to use OpenAI. For local inference, start an OpenAI-compatible server with structured output (e.g. [Ollama](https://ollama.com/) with `ollama pull qwen2.5:7b`) and set:

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
LARS_MODEL=qwen2.5:7b
```

Ratings and preferences are saved locally. A local model keeps inference on your machine; browser avatars load from Gravatar.

```sh
uv run lars-web --profile demo  # http://127.0.0.1:8000
```

- **Browser:** Install [Bun](https://bun.com/) to build the frontend automatically from source. Click the heart to like; scrolling past records a like or skip. Open **Analytics** to edit instructions and effective likes/skips in the policy grid. The chart shows stepped policy estimates and 95% credible bands for every visible option, alongside the dotted overall rolling like rate. Older profiles keep their full reward history; their policy traces start at the first saved checkpoint.
- **Terminal:** Run `uv run --locked lars --profile demo` without a frontend build. Enter `l` to like, Enter alone to skip, or `i` then Enter for stats.
- **Try it:** Use a fresh profile, rate a post, and open **Analytics** to see the update. Restart with the same profile to verify it was saved.
- **Timeline:** Open the dashboard's Timeline tab to see when options appeared, changed, or were retired. Each column follows one option: red to green is the estimated like rate, and stronger color means a narrower credible interval. Click an event to inspect its details. The local profile ledger records served posts, ratings, and policy changes.

For frontend development, run `uv run --locked lars-web` from the repo root, then `bun run dev` from `frontend/` in a second terminal. Open the Vite URL it prints. Vite proxies API requests to port `8000` by default; set `LARS_WEB_ORIGIN` for a different backend address. To refresh the production assets after editing the frontend, run `bun run build` from `frontend/`.
