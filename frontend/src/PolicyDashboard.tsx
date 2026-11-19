import { useEffect, useState, useSyncExternalStore } from 'react'
import { Link } from 'react-router'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { api, type PolicyChange, type PolicySnapshot } from './api'
import { PolicyGrid, type PolicyRow } from './PolicyGrid'
import { PolicyTimeline } from './PolicyTimeline'
import { RewardChart, policyColor } from './RewardChart'

const narrowScreen = matchMedia('(max-width: 800px)')
function subscribeToWidth(callback: () => void) {
  narrowScreen.addEventListener('change', callback)
  return () => narrowScreen.removeEventListener('change', callback)
}

export default function PolicyDashboard() {
  const [snapshot, setSnapshot] = useState<PolicySnapshot>()
  const [error, setError] = useState('')
  const [showTimeline, setShowTimeline] = useState(false)

  async function refresh() {
    setError('')
    try { setSnapshot(await api.policies()) }
    catch (error) { setError((error as Error).message) }
  }

  useEffect(() => { void refresh() }, [])

  return <main className="workspace">
    <header className="workspace-nav">
      <Link to="/">Feed</Link>
      <button type="button" aria-pressed={!showTimeline} onClick={() => setShowTimeline(false)}>Policy dashboard</button>
      <button type="button" aria-pressed={showTimeline} onClick={() => setShowTimeline(true)}>Timeline</button>
    </header>
    {error && <p className="workspace-message" role="alert">{error} <button className="control" onClick={refresh}>Retry</button></p>}
    {!snapshot && !error && <p className="workspace-message" role="status">Loading policies…</p>}
    {snapshot && <Workspace snapshot={snapshot} onChange={setSnapshot} showTimeline={showTimeline} />}
  </main>
}

function Workspace({ snapshot, onChange, showTimeline }: {
  snapshot: PolicySnapshot; onChange: (snapshot: PolicySnapshot) => void; showTimeline: boolean
}) {
  const narrow = useSyncExternalStore(subscribeToWidth, () => narrowScreen.matches)
  const [dimension, setDimension] = useState(Object.keys(snapshot.dimensions)[0])
  const [selected, setSelected] = useState<string>()
  const [includeRetired, setIncludeRetired] = useState(false)
  const [showRewards, setShowRewards] = useState(false)
  const [showRewardInterval, setShowRewardInterval] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [failed, setFailed] = useState(false)
  const { active, retired } = snapshot.dimensions[dimension]
  const focused = selected
  const posteriors = snapshot.posterior_trends[dimension] ?? {}
  const optionNames = new Set([...Object.keys(posteriors), ...active.map(option => option.text), ...retired])
  const colors = new Map(Array.from(optionNames, (text, index) => [text, policyColor(index)]))
  const counts = new Map<string, number>()
  for (const rating of snapshot.ratings) {
    const text = rating.levels[dimension]
    if (text !== undefined) counts.set(text, (counts.get(text) ?? 0) + 1)
  }
  const rows: PolicyRow[] = [
    ...active.map(option => ({ ...option, id: option.text, status: 'Active' as const,
      color: colors.get(option.text), ratings: counts.get(option.text) ?? 0 })),
    ...[...new Set(retired)].map(text => ({ id: text, text, status: 'Retired' as const,
      color: colors.get(text), ratings: counts.get(text) ?? 0 })),
    { id: '', text: '', status: 'New' },
  ]

  async function save(change: PolicyChange) {
    setBusy(true)
    setFailed(false)
    setMessage('Saving…')
    try {
      onChange(await api.changePolicy(change))
      if (change.action === 'edit' && focused === change.option) setSelected(change.text.trim())
      setMessage('Saved')
    } catch (error) {
      setFailed(true)
      setMessage((error as Error).message)
    } finally { setBusy(false) }
  }

  const dimensionPicker = <label>Dimension <select value={dimension} disabled={busy} onChange={event => {
    setDimension(event.target.value); setSelected(undefined); setMessage('')
  }}>
    {Object.keys(snapshot.dimensions).map(name => <option key={name} value={name}>{name.replaceAll('_', ' ')}</option>)}
  </select></label>

  if (showTimeline) return <section className="workspace-pane" aria-labelledby="timeline-title">
    <header className="pane-header"><h1 id="timeline-title">Policy timeline</h1>{dimensionPicker}</header>
    <PolicyTimeline key={dimension} dimension={dimension} traces={posteriors} retired={retired} />
  </section>

  return <Group orientation={narrow ? 'vertical' : 'horizontal'} className="workspace-split">
    <Panel defaultSize="58%" minSize="30%">
      <section className="workspace-pane" aria-labelledby="editor-title">
        <header className="pane-header">
          <h1 id="editor-title">Policy editor</h1>
          {dimensionPicker}
        </header>
        <PolicyGrid key={dimension} rows={rows} dimension={dimension} busy={busy} onSave={save}
          onFocusOption={setSelected} />
        <p className="pane-note">Edit instructions and effective counts in place. Enter saves, Esc cancels. New wording starts with fresh evidence.</p>
        <p className={`pane-note ${failed ? 'error' : ''}`} role={failed ? 'alert' : 'status'}>{message}</p>
      </section>
    </Panel>
    <Separator className="workspace-divider" aria-label="Resize policy and reward panels" />
    <Panel defaultSize="42%" minSize="28%">
      <section className="workspace-pane" aria-labelledby="rewards-title">
        <header className="pane-header"><h2 id="rewards-title">Reward trends</h2></header>
        <div className="chart-controls">
          <label><input type="checkbox" checked={showRewards} onChange={event => setShowRewards(event.target.checked)} /> Post rewards</label>
          <label><input type="checkbox" checked={showRewardInterval} onChange={event => setShowRewardInterval(event.target.checked)} /> Overall reward interval</label>
          <label><input type="checkbox" checked={includeRetired} onChange={event => setIncludeRetired(event.target.checked)} /> Retired options</label>
        </div>
        <RewardChart key={dimension} ratings={snapshot.ratings} posteriors={posteriors}
          focused={focused} showRewards={showRewards} showRewardInterval={showRewardInterval}
          series={rows.filter(row => row.status === 'Active' || (row.status === 'Retired' && (includeRetired || focused === row.text)))} />
      </section>
    </Panel>
  </Group>
}
