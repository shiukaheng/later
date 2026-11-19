import { expect, test } from 'bun:test'
import { rollingRewards } from '../src/rewardStats'

test('perfect and zero rates still show uncertainty with one rating', () => {
  const [liked] = rollingRewards([{ x: 1, y: 1 }])
  const [skipped] = rollingRewards([{ x: 1, y: 0 }])
  expect(liked.low).toBeCloseTo(0.2065, 4)
  expect(liked.high).toBeCloseTo(1)
  expect(skipped.low).toBeCloseTo(0)
  expect(skipped.high).toBeCloseTo(0.7935, 4)
})

test('windows use only the last 20 rewards and retain their rating index', () => {
  const points = [{ x: 1, y: 0 }, ...Array.from({ length: 20 }, (_, i) => ({ x: 4 + i * 3, y: 1 }))]
  const last = rollingRewards(points).at(-1)!
  expect(last).toMatchObject({ x: 61, y: 1, n: 20 })
  expect(last.low).toBeCloseTo(0.8389, 4)
})
