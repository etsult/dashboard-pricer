import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { RefreshCw } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Tabs } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { fetchVolTermStructure } from '@/lib/api'
import { fmtPct } from '@/lib/utils'

export default function VolSurface() {
  const [currency, setCurrency] = useState('BTC')
  const [rate, setRate] = useState(0.05)

  const { data, isLoading, error, refetch, isFetching, dataUpdatedAt } = useQuery({
    queryKey: ['vol-surface', currency, rate],
    queryFn: () => fetchVolTermStructure(currency, rate),
    staleTime: 60_000,
  })

  const ts = data?.term_structure ?? []

  const termStructureChart = [{
    x: ts.map(p => p.tenor_label),
    y: ts.map(p => p.atm_iv_pct),
    name: 'Spot ATM IV',
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    line: { color: '#3b82f6', width: 2 },
    marker: { size: 7, color: ts.map(p => p.is_calendar_arb ? '#ef4444' : '#3b82f6') },
  }, {
    x: ts.filter(p => p.fwd_vol_pct != null).map(p => p.fwd_label ?? ''),
    y: ts.filter(p => p.fwd_vol_pct != null).map(p => p.fwd_vol_pct!),
    name: 'Forward Vol',
    type: 'bar' as const,
    marker: { color: '#a78bfa', opacity: 0.7 },
    yaxis: 'y2',
  }]

  const totalVarChart = [{
    x: ts.map(p => p.days),
    y: ts.map(p => p.total_var),
    name: 'Total Variance w(T)',
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    fill: 'tozeroy' as const,
    line: { color: '#22c55e', width: 2 },
    marker: { size: 6, color: ts.map(p => p.is_calendar_arb ? '#ef4444' : '#22c55e') },
  }]

  const updatedAt = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : '—'
  const front = ts[0]?.atm_iv_pct
  const back = ts[ts.length - 1]?.atm_iv_pct
  const slope = front && back ? back - front : null

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Vol Surface</h1>
          <p className="text-muted text-sm mt-1">Live crypto option term structure from Deribit</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw size={14} className={isFetching ? 'animate-spin' : ''} />
          Refresh
        </Button>
      </div>

      <Card>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Select label="Currency" value={currency} onChange={e => setCurrency(e.target.value)}
            options={[{ value: 'BTC', label: 'BTC' }, { value: 'ETH', label: 'ETH' }, { value: 'SOL', label: 'SOL' }]} />
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted uppercase tracking-wide">Rate</label>
            <input type="number" value={rate} step="0.01" className="h-9 rounded-md border border-border bg-panel px-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-accent"
              onChange={e => setRate(+e.target.value)} />
          </div>
          <div className="col-span-2 flex items-end">
            <p className="text-xs text-muted">Last updated: <span className="text-slate-400">{updatedAt}</span></p>
          </div>
        </div>
      </Card>

      {isLoading && <p className="text-muted text-center py-12">Loading term structure…</p>}
      {error && <p className="text-negative text-center py-8">Error fetching data. Is the API running?</p>}

      {data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <Stat label="Spot" value={`$${data.spot.toLocaleString()}`} />
            <Stat label="Quotes" value={data.n_quotes} />
            <Stat label="Front ATM IV" value={fmtPct(front)} />
            <Stat label="Back ATM IV"  value={fmtPct(back)} />
            <Stat label="Arb Alerts" value={data.n_arb_violations}
              color={data.n_arb_violations > 0 ? 'text-negative' : 'text-positive'} />
          </div>

          {slope !== null && (
            <div className="flex gap-2 items-center">
              <Badge variant={slope > 0 ? 'warning' : 'positive'}>
                {slope > 0 ? 'Contango' : 'Backwardation'} {Math.abs(slope).toFixed(1)}%
              </Badge>
              {data.n_arb_violations > 0 && (
                <Badge variant="negative">{data.n_arb_violations} Calendar Arb!</Badge>
              )}
            </div>
          )}

          <Tabs tabs={[{ id: 'ts', label: 'Term Structure' }, { id: 'var', label: 'Total Variance' }]}>
            {active => (
              <Card>
                {active === 'ts' && (
                  <PlotlyChart
                    data={termStructureChart}
                    layout={{
                      title: { text: `${currency} Vol Term Structure` },
                      yaxis: { title: { text: 'Spot ATM IV (%)' } },
                      yaxis2: { title: { text: 'Forward Vol (%)' }, overlaying: 'y', side: 'right' },
                      barmode: 'overlay',
                    }}
                  />
                )}
                {active === 'var' && (
                  <PlotlyChart
                    data={totalVarChart}
                    layout={{
                      title: { text: 'Total Variance w(T) — must be non-decreasing' },
                      xaxis: { title: { text: 'Days to Expiry' } },
                      yaxis: { title: { text: 'w(T) = σ²·T' } },
                    }}
                  />
                )}
              </Card>
            )}
          </Tabs>

          {/* Term structure table */}
          <Card>
            <CardHeader><CardTitle>Term Structure Data</CardTitle></CardHeader>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead><tr className="border-b border-border">
                  {['Tenor', 'Days', 'ATM IV', 'Fwd Vol', 'Total Var', 'Arb?'].map(h => (
                    <th key={h} className="text-left px-3 py-2 text-muted uppercase tracking-wide font-medium">{h}</th>
                  ))}
                </tr></thead>
                <tbody>
                  {ts.map((p, i) => (
                    <tr key={i} className="border-b border-border/40 hover:bg-panel/50">
                      <td className="px-3 py-2 font-mono text-slate-300">{p.tenor_label}</td>
                      <td className="px-3 py-2 font-mono">{p.days.toFixed(0)}</td>
                      <td className="px-3 py-2 font-mono text-blue-400">{fmtPct(p.atm_iv_pct)}</td>
                      <td className="px-3 py-2 font-mono text-purple-400">{p.fwd_vol_pct != null ? fmtPct(p.fwd_vol_pct) : '—'}</td>
                      <td className="px-3 py-2 font-mono">{p.total_var.toFixed(4)}</td>
                      <td className="px-3 py-2">
                        {p.is_calendar_arb
                          ? <Badge variant="negative">ARB</Badge>
                          : <Badge variant="positive">OK</Badge>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
