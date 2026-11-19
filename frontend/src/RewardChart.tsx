import {
  Chart as ChartJS, Filler, Legend, LinearScale, LineElement, PointElement, Tooltip,
  type ChartDataset, type ChartOptions,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import type { PosteriorPoint, Rating } from './api'
import { REWARD_WINDOW, rollingRewards, type RewardPoint } from './rewardStats'

ChartJS.register(Filler, Legend, LinearScale, LineElement, PointElement, Tooltip)

type TrendPoint = RewardPoint & { n?: number; low?: number; high?: number }
type Series = ChartDataset<'line', TrendPoint[]> & { id: string; bandFor?: string }
const percent = (value: number) => `${Math.round(value * 100)}%`
const precisePercent = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 })
export function policyColor(index: number) {
  // Space neighboring options apart in hue, even when their wording is similar.
  return `hsl(${(index * 137.508 + 225) % 360} 65% 48%)`
}

// In Chart.js, 'before' holds the previous y until the next point's x.
const holdUntilUpdate = 'before'

function uncertaintyBand(trend: Series, color: string): Series[] {
  const first = trend.data[0]
  if (first && trend.data.every(point => point.x === first.x)) {
    return [{
      id: `${trend.id}:range`, bandFor: trend.id, order: 1,
      data: trend.data.flatMap(({ x, low, high }) => [{ x, y: low! }, { x, y: high! }]),
      borderColor: color, borderWidth: 5, pointRadius: 0,
    }]
  }
  return (['low', 'high'] as const).map(bound => ({
    id: `${trend.id}:${bound}`, bandFor: trend.id, order: 1,
    data: trend.data.map(point => ({ x: point.x, y: point[bound]! })),
    fill: bound === 'high' ? '-1' : false,
    stepped: trend.stepped, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0,
    backgroundColor: color,
  }))
}

const options: ChartOptions<'line'> = {
  responsive: true, maintainAspectRatio: false, animation: false,
  interaction: { mode: 'nearest', axis: 'x', intersect: false },
  scales: {
    x: { type: 'linear', title: { display: true, text: 'Rating #'}, ticks: { precision: 0 } },
    y: { min: 0, max: 1, ticks: { callback: value => percent(Number(value)) } },
  },
  plugins: {
    legend: {
      labels: { boxWidth: 10, filter: (item, data) => !(data.datasets[item.datasetIndex!] as Series).bandFor },
      onClick: (_, item, legend) => {
        const { chart } = legend
        const index = item.datasetIndex!
        const visible = !chart.isDatasetVisible(index)
        const id = (chart.data.datasets[index] as Series).id
        chart.data.datasets.forEach((dataset, i) => {
          if (i === index || (dataset as Series).bandFor === id) chart.setDatasetVisibility(i, visible)
        })
        chart.update()
      },
    },
  },
}

export function RewardChart({ ratings, posteriors, series, focused, showRewards, showRewardInterval }: {
  ratings: Rating[]; posteriors: Record<string, PosteriorPoint[]>; series: { text: string; color?: string }[];
  focused?: string; showRewards: boolean; showRewardInterval: boolean
}) {
  const overall = ratings.map((rating, index) => ({ x: index + 1, y: rating.reward }))
  const overallTrend: Series = {
    id: 'overall', label: 'Overall observed like rate', data: rollingRewards(overall), borderDash: [1, 5], borderCapStyle: 'round',
    stepped: holdUntilUpdate, borderColor: '#8994a4', backgroundColor: '#8994a4', pointRadius: ratings.length === 1 ? 3 : 0,
  }
  const datasets = overall.length ? [overallTrend] : []
  if (showRewardInterval && overall.length) datasets.push(...uncertaintyBand(overallTrend, '#8994a438'))
  if (showRewards && overall.length) datasets.push({ id: 'rewards', label: 'Post reward', data: overall, showLine: false, backgroundColor: '#8994a4', pointRadius: 2 })
  for (const { text, color } of series) {
    const points = posteriors[text]
    if (!points?.length) continue
    const trend: Series = {
      id: `option:${text}`, label: text,
      data: points.map(({ rating, estimate, credible_interval: [low, high] }) => ({ x: rating, y: estimate, low, high })),
      stepped: holdUntilUpdate, borderColor: color, backgroundColor: color,
      borderWidth: focused === text ? 3 : 2, pointRadius: points.length === 1 ? 3 : 0,
    }
    datasets.push(trend)
    const opacity = focused === text ? '20%' : '11%'
    datasets.push(...uncertaintyBand(trend, `color-mix(in srgb, ${color ?? '#8994a4'} ${opacity}, transparent)`))
  }

  const chartOptions: ChartOptions<'line'> = {
    ...options,
    plugins: {
      ...options.plugins,
      tooltip: {
        filter: item => !(item.dataset as Series).bandFor,
        callbacks: { label: ({ dataset, raw }) => {
          const { y, n, low, high } = raw as TrendPoint
          const label = `${dataset.label}: ${precisePercent.format(y)}`
          if (low === undefined || high === undefined) return label
          const interval = `${precisePercent.format(low)}–${precisePercent.format(high)}`
          if (n === undefined) return `${label} (95% credible interval: ${interval})`
          return showRewardInterval ? `${label} (95% Wilson interval: ${interval}, n=${n})` : label
        } },
      },
    },
  }
  return <>
    {datasets.length ? <div className="chart">
      <Line data={{ datasets }} datasetIdKey="id" options={chartOptions} role="img" aria-label="Option trends and overall like rate" />
    </div> : <p className="pane-note">No recorded trends yet.</p>}
    <p className="pane-note">Each colored step shows a policy estimate and its 95% credible band. Use the legend to show or hide them together. The dotted line shows the observed like rate over the last {REWARD_WINDOW} ratings.</p>
    {showRewardInterval && <p className="pane-note">The gray band is an approximate 95% Wilson interval for recorded rewards.</p>}
  </>
}
