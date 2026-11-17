# LARS

Run the browser feed with `uv run lars-web` and open http://localhost:8000, or run
the terminal feed with `uv run lars`.

## Profiles

Use `--profile` to keep separate preferences and rating histories:

```sh
uv run lars-web --profile work
uv run lars --profile work
uv run lars-web --profile personal
```

Both commands use the same profile for the same name. The default profile uses the
existing `data/session.json`; named profiles use `data/profiles/<name>.json` and
start fresh. Names may contain letters, numbers, underscores and hyphens, and
must start with a letter or number. Profiles share the synthetic author pool.
To run browser profiles simultaneously, give each a different port, for example
`LARS_PORT=8001 uv run lars-web --profile work`.

LLM temperature defaults to 1.2. Set `LARS_TEMPERATURE` to adjust the variety,
for example `LARS_TEMPERATURE=1.4 uv run lars-web` (subject to your model's
supported range).

Ratings gradually lose weight as preferences change. `LARS_DISCOUNT` controls
the per-rating decay (default `0.995`, a half-life of about 138 ratings). Set
it to `1` to disable decay. The info view shows effective, discounted sample
sizes; retirement also uses these effective counts. Existing saved counts start
decaying with the next rating. Retired options are not automatically restored.

The browser server binds to localhost by default. To make it reachable on your
LAN and Tailscale interfaces, run `uv run lars-web --host 0.0.0.0` and open
`http://<machine-ip>:8000` from another device. Alternatively, bind to your
Tailscale IP with `--host <tailscale-ip>`. The feed has no authentication; anyone
who can reach the server can view and rate posts in the selected profile.

## Current events

For news context, get a free token from [The News API](https://www.thenewsapi.com/)
and add `THENEWSAPI_TOKEN=your-token` to your local `.env`, or set it in the
environment before launching LARS. The free plan's top-stories endpoint returns
up to three articles per request. On first use LARS fetches up to 10 pages to
prefill its cache, then refreshes it with one request every 30 minutes across
local processes. It caches up to 30 recent stories in `data/news.json` and uses
articles no older than 48 hours. A continuously running installation makes up to
10 prefill requests plus about 47 scheduled requests in its first day; other
applications using the same token consume their own quota.

`current_events` is a normal evolving bandit dimension, initially offering a
news-based and an evergreen option. Likes and skips guide which options are
sampled, and evolution can propose new ones; there is no fixed news percentage.
The proposal and generation models can choose to call a news-cache tool; the
cached headlines are supplied only when they call it. Sources are recorded with
ratings but not shown in the feed. Source selection depends on the model and is not guaranteed to be
perfect. If the token or recent stories are unavailable, the feed continues
without selecting or crediting this dimension. Existing profiles acquire the
new dimension automatically.
