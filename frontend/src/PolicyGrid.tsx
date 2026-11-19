import { useMemo } from 'react'
import {
  AgGridProvider, AgGridReact,
} from 'ag-grid-react'
import {
  ClientSideRowModelModule, NumberEditorModule, SelectEditorModule, TextEditorModule, TooltipModule,
  themeQuartz, colorSchemeVariable,
  type CellEditRequestEvent, type ColDef,
} from 'ag-grid-community'
import type { PolicyChange } from './api'

export interface PolicyRow {
  id: string
  text: string
  status: 'Active' | 'Retired' | 'New'
  color?: string
  likes?: number
  skips?: number
  estimate?: number
  effective_n?: number
  ratings?: number
}

const modules = [ClientSideRowModelModule, NumberEditorModule, SelectEditorModule, TextEditorModule, TooltipModule]
const theme = themeQuartz.withPart(colorSchemeVariable).withParams({ spacing: 5, fontSize: 13, borderRadius: 0 })
const effective = new Intl.NumberFormat('en', { maximumFractionDigits: 3 })
const percent = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 })

export function PolicyGrid({ rows, dimension, busy, onSave, onFocusOption }: {
  rows: PolicyRow[]; dimension: string; busy: boolean
  onSave: (change: PolicyChange) => Promise<void>; onFocusOption: (option: string) => void
}) {
  const activeCount = rows.filter(row => row.status === 'Active').length
  const columns = useMemo<ColDef<PolicyRow>[]>(() => {
    const numberColumn = (field: 'likes' | 'skips', headerName: string): ColDef<PolicyRow> => ({
      field, headerName, width: 88, cellEditor: 'agNumberCellEditor',
      cellEditorParams: { min: 0, max: 1e12, step: 'any' },
      editable: ({ data }) => !busy && data?.status === 'Active',
      valueFormatter: ({ value }) => value == null ? '' : effective.format(value),
    })
    return [
      { colId: 'trend', headerName: '', width: 34, minWidth: 34, resizable: false,
        cellRenderer: ({ data }: { data?: PolicyRow }) => data?.color && <button type="button" className="policy-swatch"
          aria-label={`Focus trend for ${data.text}`} title="Focus trend" onClick={() => onFocusOption(data.text)}
          style={{ backgroundColor: data.color }} /> },
      { field: 'text', headerName: 'Instruction', flex: 1, minWidth: 175, tooltipField: 'text',
        editable: ({ data }) => !busy && data?.status !== 'Retired',
        valueFormatter: ({ data, value }) => data?.status === 'New' ? '+ New instruction' : value },
      { field: 'status', width: 80, editable: ({ data }) => !busy && activeCount > 1 && data?.status === 'Active',
        cellEditor: 'agSelectCellEditor', cellEditorParams: { values: ['Active', 'Retired'] },
        valueFormatter: ({ value }) => value === 'New' ? '' : value },
      numberColumn('likes', 'Eff. likes'), numberColumn('skips', 'Eff. skips'),
      { field: 'estimate', width: 85, valueFormatter: ({ value }) => value == null ? '' : percent.format(value) },
      { field: 'effective_n', headerName: 'Eff. n', width: 68,
        valueFormatter: ({ value }) => value == null ? '' : effective.format(value) },
      { field: 'ratings', headerName: 'Rated', width: 65 },
    ]
  }, [activeCount, busy, onFocusOption])

  function commit({ data, colDef, newValue }: CellEditRequestEvent<PolicyRow>) {
    if (!data || busy) return
    const field = colDef.field
    if (field === 'text' && typeof newValue === 'string' && newValue.trim() && newValue.trim() !== data.text) {
      void onSave(data.status === 'New' ? { action: 'add', dimension, text: newValue }
        : { action: 'edit', dimension, option: data.text, text: newValue })
    } else if ((field === 'likes' || field === 'skips') && typeof newValue === 'number' && newValue !== data[field]) {
      void onSave({ action: 'evidence', dimension, option: data.text, field, value: newValue })
    } else if (field === 'status' && newValue === 'Retired') {
      void onSave({ action: 'retire', dimension, option: data.text })
    }
  }

  return <div className="policy-grid" data-ag-theme-mode={matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'}>
    <AgGridProvider modules={modules}>
      <AgGridReact theme={theme} rowData={rows} columnDefs={columns} getRowId={({ data }) => data.status === 'New' ? 'new' : `${data.status}:${data.id}`}
        readOnlyEdit singleClickEdit stopEditingWhenCellsLoseFocus onCellEditRequest={commit}
        rowHeight={31} headerHeight={34} defaultColDef={{ sortable: false }} />
    </AgGridProvider>
  </div>
}
