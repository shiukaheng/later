import { useEffect, useMemo, useState } from 'react'
import { AgGridProvider, AgGridReact } from 'ag-grid-react'
import { CellStyleModule, ClientSideRowModelModule, PaginationModule, TooltipModule, themeQuartz, colorSchemeVariable,
  type CellStyle, type ColDef } from 'ag-grid-community'
import { api, type LedgerEvent, type PosteriorPoint } from './api'

const modules = [CellStyleModule, ClientSideRowModelModule, PaginationModule, TooltipModule]
const theme = themeQuartz.withPart(colorSchemeVariable).withParams({ spacing: 5, fontSize: 13, borderRadius: 0 })
const percent = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 })

type Cell = { point?: PosteriorPoint; status?: 'added' | 'retired' | 'inactive' }
type Row = { entry: LedgerEvent; cells: Record<string, Cell> }

function eventLabel(entry: LedgerEvent) {
  switch (entry.kind) {
    case 'rated': return entry.details.reward ? 'Like' : 'Skip'
    case 'served': return 'Post served'
    case 'override': return 'Evidence override'
    case 'retired': return entry.details.reason === 'automatic' ? 'Automatic retirement' : 'Manual retirement'
    case 'added': return 'Option added'
    case 'edited': return 'Option replaced'
    case 'initialized': return 'Initialized'
    default: return 'Checkpoint'
  }
}

function matrix(events: LedgerEvent[], dimension: string, traces: Record<string, PosteriorPoint[]>, showPosts: boolean) {
  const active = new Map<string, PosteriorPoint>()
  const ended = new Set<string>()
  const offsets = new Map<string, number>()
  return events.flatMap(entry => {
    const added = new Set(entry.added.filter(([dim]) => dim === dimension).map(([, text]) => text))
    const removed = new Set(entry.removed.filter(([dim]) => dim === dimension).map(([, text]) => text))
    for (const [text, points] of Object.entries(traces)) {
      let i = offsets.get(text) ?? 0
      while (i < points.length && points[i].event <= entry.event) active.set(text, points[i++])
      offsets.set(text, i)
    }
    const record = entry.kind === 'served' || entry.kind === 'rated'
    const relevant = entry.details.dimension === dimension || added.size || removed.size || !entry.details.dimension && !record
    const rows: Row[] = []
    if (record ? showPosts : relevant) rows.push({ entry, cells: Object.fromEntries(Object.keys(traces).map(text => [text, {
      point: active.get(text),
      status: removed.has(text) ? 'retired' : added.has(text) ? 'added' : ended.has(text) ? 'inactive' : undefined,
    }])) })
    for (const text of removed) { active.delete(text); ended.add(text) }
    for (const text of added) ended.delete(text)
    return rows
  }).reverse()
}

export function PolicyTimeline({ dimension, traces, retired }: {
  dimension: string; traces: Record<string, PosteriorPoint[]>; retired: string[]
}) {
  const [events, setEvents] = useState<LedgerEvent[]>()
  const [showPosts, setShowPosts] = useState(false)
  const [selected, setSelected] = useState<LedgerEvent>()
  const [error, setError] = useState('')

  useEffect(() => {
    api.timeline().then(setEvents, error => setError(error.message))
  }, [])

  const columns = useMemo<ColDef<Row>[]>(() => [
    { headerName: 'Event', pinned: 'left', width: 192, valueGetter: ({ data }) => data
      ? `#${data.entry.event + 1} · ${eventLabel(data.entry)}` : '' },
    { headerName: 'Rating', width: 82, valueGetter: ({ data }) => data?.entry.rating },
    ...Array.from(new Set([...Object.keys(traces), ...retired]), (text): ColDef<Row> => ({
      colId: `option:${text}`, headerName: text, headerTooltip: text, width: 155,
      valueGetter: ({ data }) => data?.cells[text],
      valueFormatter: ({ value }: { value?: Cell }) => {
        const rate = value?.point ? percent.format(value.point.estimate) : ''
        if (value?.status === 'added') return `+ ${rate}`
        if (value?.status === 'retired') return `Retired ${rate}`
        if (value?.status === 'inactive') return '—'
        return rate
      },
      cellStyle: ({ value }: { value?: Cell }): CellStyle => {
        if (!value?.point) return { opacity: value?.status === 'inactive' ? 0.45 : 1, backgroundColor: 'transparent' }
        const { estimate, credible_interval: [low, high] } = value.point
        const opacity = 0.15 + 0.75 * (1 - (high - low))
        return { opacity: 1, backgroundColor: `hsl(${estimate * 125} 75% 62% / ${opacity})` }
      },
      tooltipValueGetter: ({ value, data }) => value?.point
        ? `${text}: ${percent.format(value.point.estimate)} · 95% credible interval ${value.point.credible_interval.map((bound: number) => percent.format(bound)).join('–')} · effective n ${value.point.effective_n.toFixed(2)}`
        : data?.cells[text]?.status === 'inactive' ? `${text}: retired` : undefined,
    })),
  ], [traces, retired])
  const rows = useMemo(() => matrix(events ?? [], dimension, traces, showPosts), [events, dimension, traces, showPosts])
  return <>
    <div className="chart-controls">
      <label><input type="checkbox" checked={showPosts} onChange={event => setShowPosts(event.target.checked)} /> Show posts and ratings</label>
      <span>Red → green: estimated like rate. More opaque: narrower 95% credible interval.</span>
    </div>
    {error ? <p className="pane-note error" role="alert">{error}</p> : events ?
      <div className="policy-grid" data-ag-theme-mode={matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'}>
        <AgGridProvider modules={modules}>
          <AgGridReact theme={theme} rowData={rows} columnDefs={columns} rowHeight={31}
            getRowId={({ data }) => `${data.entry.event}`} onCellClicked={({ data }) => setSelected(data?.entry)}
            pagination paginationPageSize={25} paginationPageSizeSelector={[25, 50, 100]} defaultColDef={{ sortable: false }} />
        </AgGridProvider>
      </div> : <p className="pane-note" role="status">Loading timeline…</p>}
    {selected && <EventDetails entry={selected} />}
  </>
}

function EventDetails({ entry }: { entry: LedgerEvent }) {
  const d = entry.details
  return <section className="event-details" aria-label="Event details">
    <strong>{eventLabel(entry)}</strong> · Rating {entry.rating} · {entry.at ? new Date(entry.at).toLocaleString() : 'Time unknown'}
    {d.option && <p>{d.option}{d.replacement ? ` → ${d.replacement}` : ''}</p>}
    {d.origin && <p>Source: {d.origin}</p>}
    {d.reason === 'automatic' && d.probability !== undefined && d.threshold !== undefined &&
      <p>{percent.format(d.probability)} probability of underperforming “{d.comparison}” (threshold {percent.format(d.threshold)}; effective samples {d.effective_n?.toFixed(1)} vs {d.comparison_n?.toFixed(1)}).</p>}
    {d.field && <p>Effective {d.field}: {d.before} → {d.after}</p>}
    {d.post_id != null && <p>Post #{d.post_id}{d.levels && ` · ${Object.values(d.levels).join(' / ')}`}</p>}
    {entry.kind === 'checkpoint' && entry.removed.length > 0 && <p>Retirement reason unknown.</p>}
  </section>
}
