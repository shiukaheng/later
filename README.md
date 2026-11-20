# LATER: Language Arms for Thompson-based Evolving Recommendation

<img src="feed-preview.png" alt="Light-mode LATER For You feed on a gray background, showing six generated tweets" width="560">

**What if users could own their recommendation systems, audit why they suggest content, and write the rules those systems explore?**

LATER is a self-hostable proof of concept for personalization from sparse feedback. The demo is a synthetic Twitter/X feed that starts without an account or imported history. **The LLM proposes and follows readable instructions; a bandit tests which instructions work using likes and skips.**

## Inspect and edit your feed

The **Analytics** dashboard is central to the experiment: can users see what their recommender has learned and change the rules it explores?

<img src="dashboard-preview.png" alt="LATER Analytics dashboard with an editable policy grid on the left and estimated like rates with credible bands on the right" width="900">

- **Edit the rules.** Add, rewrite, or retire instructions, and override the effective likes and skips that guide future recommendations.
- **See what the system knows.** Compare estimated like rates and 95% credible bands over time against the overall observed like rate.
- **Audit changes.** The timeline shows which posts were served and rated, how the estimates changed, and why policies were retired. Overrides do not rewrite past ratings.

## How it works

- **Multi-armed bandit on readable suggestion policies.** [Thompson sampling](https://proceedings.neurips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) selects instructions for the LLM to follow. Feedback on the generated post updates the bandit. The LLM proposes new options as the bandit retires weak ones, without fine-tuning the model.
- **Factored action space.** Each post combines a topic, style, visual format, and mood. One rating updates each selected option, sharing evidence across combinations without modeling interactions.

LATER combines existing techniques to generate posts from immediate binary feedback. It does not rank existing posts, optimize long-term outcomes, or establish how many ratings are enough.

## Related research

- [Factored Bandits](https://arxiv.org/abs/1807.01488) (2018) studies compositional actions under limited feedback.
- [GAMBITTS](https://arxiv.org/abs/2505.16311) and [Prompt Optimization with Logged Bandit Data](https://arxiv.org/abs/2504.02646) (2025) study feedback mediated by generated outputs. LATER uses simpler online updates that credit the selected instructions.
- [Promptbreeder](https://proceedings.mlr.press/v235/fernando24a.html) (2024) evolves prompts through iterative evaluation. [Bonsai](https://doi.org/10.1145/3772318.3791855) (2026) lets users shape their social feeds through natural-language intentions.

## Run the web demo

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Bun](https://bun.com/), then clone the repo. On first launch, `uv` sets up Python 3.14 and the backend dependencies, and Bun builds the web UI automatically.

```sh
git clone https://github.com/shiukaheng/later.git
cd later
```

Choose a model setup and create a `.env` file in the repo root:

- **OpenAI:** Set `OPENAI_API_KEY=your-key`. The default model is `gpt-4o-mini`.
- **Local:** Install and start [Ollama](https://ollama.com/) (`ollama serve` if it isn't already running). Run `ollama pull qwen2.5:7b` in another terminal, then put the following in `.env`:

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
LATER_MODEL=qwen2.5:7b
```

Ratings and preferences are saved locally. A local model keeps inference on your machine; browser avatars load from Gravatar.

Launch the demo from the repo root:

```sh
uv run later-web --profile demo
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Click a heart, then scroll past the post to record your like. Scrolling past without liking records a skip. Open **Analytics** to inspect the updated estimates or edit the policies. Use an unused profile name to see the feed learn from scratch.

Optional terminal demo: `uv run later --profile demo` works without Bun. Type `l` then Enter to like, Enter alone to skip, or `i` then Enter for stats.

For frontend development, run `uv run later-web` from the repo root and `bun install --frozen-lockfile && bun run dev` from `frontend/` in a second terminal. Open the Vite URL it prints. Set `LATER_WEB_ORIGIN` if the backend uses a port other than `8000`.
