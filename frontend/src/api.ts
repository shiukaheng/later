export type Reward = 0 | 1

export interface Post {
  id: number
  text: string
  author: { name: string; handle: string; avatar: string }
  stats: { likes: number; minutes_ago: number }
}

export interface Rating {
  reward: Reward
  levels: Record<string, string>
}

export interface PosteriorPoint {
  rating: number
  event: number
  estimate: number
  credible_interval: [number, number]
  effective_n: number
}

export interface LedgerEvent {
  event: number
  rating: number
  kind: string
  at: string | null
  details: {
    dimension?: string; option?: string; replacement?: string | null; reason?: string;
    probability?: number; threshold?: number; comparison?: string; effective_n?: number;
    comparison_n?: number; origin?: string; field?: string; before?: number; after?: number;
    reward?: Reward; post_id?: number | null; levels?: Record<string, string>;
  }
  added: [string, string][]
  removed: [string, string][]
}

export interface PolicySnapshot {
  dimensions: Record<string, {
    active: {
      text: string; estimate: number; effective_n: number; likes: number; skips: number;
    }[]
    retired: string[]
  }>
  ratings: Rating[]
  posterior_trends: Record<string, Record<string, PosteriorPoint[]>>
}

export type PolicyChange = { dimension: string } & (
  | { action: 'add'; text: string }
  | { action: 'edit'; option: string; text: string }
  | { action: 'retire'; option: string }
  | { action: 'evidence'; option: string; field: 'likes' | 'skips'; value: number }
)

async function request<T>(path: string, body?: object): Promise<T> {
  const response = await fetch(path, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : undefined)
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.error ?? `Request failed (${response.status})`)
  }
  return response.json()
}

export const api = {
  next: () => request<Post>('/next'),
  rate: (id: number, reward: Reward) => request<object>('/rate', { id, reward }),
  policies: () => request<PolicySnapshot>('/api/policies'),
  timeline: () => request<LedgerEvent[]>('/api/policies/timeline'),
  changePolicy: (change: PolicyChange) => request<PolicySnapshot>('/api/policies', change),
}
