export const REWARD_WINDOW = 20

export type RewardPoint = { x: number; y: number }

export function rollingRewards(points: RewardPoint[]) {
  const z = 1.96
  let likes = 0
  return points.map(({ x, y }, index) => {
    likes += y - (index >= REWARD_WINDOW ? points[index - REWARD_WINDOW].y : 0)
    const n = Math.min(index + 1, REWARD_WINDOW)
    const rate = likes / n
    const scale = 1 + z * z / n
    const center = (rate + z * z / (2 * n)) / scale
    const margin = z * Math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / scale
    return { x, y: rate, n, low: Math.max(0, center - margin), high: Math.min(1, center + margin) }
  })
}
