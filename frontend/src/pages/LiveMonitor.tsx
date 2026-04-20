import { useState, useCallback } from 'react'
import { Radio, Wifi, WifiOff, RefreshCw } from 'lucide-react'
import { Card, Stat } from '@/components/ui/card'
import { Select } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs } from '@/components/ui/tabs'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { useWebSocket } from '@/hooks/useWebSocket'
import type { VolTermStructureResponse, VolPoint } from '@/lib/api'
import { fmtPct } from '@/lib/utils'

const intervalOptions = [
  { value: '10',  label: '10 seconds' },
  { value: '30',  label: '30 seconds' },
  { value: '60',  label: '1 minute'   },
  { value: '300', label: '5 minutes'  },
]

function StatusDot({ status }: { status: string }) {
  if (status === 'open')        return <span className="flex items-center gap-1.5 text-positive text-xs"><span className="w-2 h-2 rounded-full bg-positive animate-pulse" />Live</span>
  if (status === 'connecting')  return <span className="flex items-center gap-1.5 text-warning text-xs"><span className="w-2 h-2 rounded-full bg-warning animate-pulse" />Connecting…</span>
  if (status === 'error')       return <span className="flex items-center gap-1.5 text-negative text-xs"><span className="w-2 h-2 rounded-full bg-negative" />Error</span>
  return <span className="flex items-center gap-1.5 text-muted text-xs"><span className="w-2 h-2 rounded-full bg-muted" />Disconnected</span>
}

export default function LiveMonitor() {
  const [currency, setCurrency] = useState('BTC')
  const [interval, setInterval] = useState('30')
  const [enabled, setEnabled] = useState(true)
  const [data, setData] = useState<VolTermStructureResponse | null>(null)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  const wsUrl = `/ws/live-monitor?currency=${currency}&interval=${interval}`

  const onMessage = useCallback((msg: VolTermStructureResponse | { error: string }) => {
    if ('error' in msg) {
      setError(msg.error)
    } else {
      setData(msg)
      setError(null)
      setLastUpdate(new Date().toLocaleTimeString())
    }
  }, [])

  const { status, reconnect } = useWebSocket<VolTermStructureResponse | { error: string }>(
    wsUrl, { onMessage, enabled }
  )

  const ts: VolPoint[] = data?.term_structure ?? []
  const front = ts[0]?.atm_iv_pct
  const back  = ts[ts.length - 1]?.atm_iv_pct
  const slope = front && back ? back - front : null
  const arbCount = ts.filter(p => p.is_calendar_arb).length

  const termStructureChart = [{
    x: ts.map(p => p.tenor_label),
    y: ts.map(p => p.atm_iv_pct),
    name: 'ATM IV',
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    line: { color: '#3b82f6', width: 2 },
    marker: { size: 8, color: ts.map(p => p.is_calendar_arb ? '#ef4444' : '#3b82f6') },
  }]

  const fwdVolChart = [{
    x: ts.filter(p => p.fwd_vol_pct != null).map(p => p.fwd_label ?? ''),
    y: ts.filter(p => p.fwd_vol_pct != null).map(p => p.fwd_vol_pct!),
    name: 'Fwd Vol',
    type: 'bar' as const,
    marker: {
      color: ts.filter(p => p.fwd_vol_pct != null).map(p => p.is_calendar_arb ? '#ef4444' : '#a78bfa'),
    },
  }]

  const totalVarChart = [{
    x: ts.map(p => p.days),
    y: ts.map(p => p.total_var),
    name: 'w(T)',
    type: 'scatter' as const,
    mode: 'lines+markers' as const,
    fill: 'tozeroy' as const,
    line: { color: '#22c55e', width: 2 },
  }]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Radio size={20} className="text-accent" />
          <div>
            <h1 className="text-xl font-bold text-white">Live Vol Monitor</h1>
            <p className="text-muted text-sm">Real-time crypto options term structure via WebSocket</p>
          </div>
        </div>
        <StatusDot status={status} />
      </div>

      {/* Controls */}
      <Card>
        <div className="flex flex-wrap gap-4 items-end">
          <Select label="Currency" value={currency}
            onChange={e => { setCurrency(e.target.value); setData(null) }}
            options={[{ value: 'BTC', label: 'BTC' }, { value: 'ETH', label: 'ETH' }, { value: 'SOL', label: 'SOL' }]}
          />
          <Select label="Refresh Interval" value={interval}
            onChange={e => setInterval(e.target.value)}
            options={intervalOptions}
          />
          <Button
            variant={enabled ? 'danger' : 'positive'}
            size="md"
            onClick={() => setEnabled(e => !e)}
          >
            {enabled ? <><WifiOff size={14} /> Disconnect</> : <><Wifi size={14} /> Connect</>}
          </Button>
          {!enabled && (
            <Button variant="outline" size="md" onClick={() => { setEnabled(true); reconnect() }}>
              <RefreshCw size={14} /> Reconnect
            </Button>
          )}
          {lastUpdate && <p className="text-xs text-muted self-end">Last update: {lastUpdate}</p>}
        </div>
      </Card>

      {error && (
        <div className="bg-negative/10 border border-negative/30 rounded-lg p-3 text-negative text-sm">
          {error}
        </div>
      )}

      {!data && status === 'connecting' && (
        <p className="text-muted text-center py-12">Connecting to Deribit via WebSocket…</p>
      )}

      {data && (
        <>
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <Stat label="Spot" value={`$${data.spot.toLocaleString()}`} />
            <Stat label="Quotes" value={data.n_quotes} sub={`at ${data.fetched_at}`} />
            <Stat label="Front ATM IV" value={fmtPct(front)} />
            <Stat label="Back ATM IV"  value={fmtPct(back)} />
            <Stat label="Arb Alerts" value={arbCount}
              color={arbCount > 0 ? 'text-negative' : 'text-positive'} />
          </div>

          {/* Slope badge */}
          <div className="flex gap-2">
            {slope !== null && (
              <Badge variant={slope > 0 ? 'warning' : 'positive'}>
                {slope > 0 ? 'Contango' : 'Backwardation'} {Math.abs(slope).toFixed(1)} vol pts
              </Badge>
            )}
            {arbCount > 0 && <Badge variant="negative">{arbCount} Calendar Arb Violation{arbCount > 1 ? 's' : ''}!</Badge>}
          </div>

          {/* Charts */}
          <Tabs tabs={[
            { id: 'ts',      label: 'Term Structure' },
            { id: 'fwdvol',  label: 'Forward Vol Strip' },
            { id: 'arb',     label: 'Arb Scanner' },
            { id: 'totalvar',label: 'Total Variance' },
          ]}>
            {active => (
              <Card>
                {active === 'ts' && (
                  <PlotlyChart
                    data={termStructureChart}
                    layout={{
                      title: { text: `${currency} ATM Implied Vol Term Structure` },
                      yaxis: { title: { text: 'ATM IV (%)' } },
                    }}
                  />
                )}
                {active === 'fwdvol' && (
                  <PlotlyChart
                    data={fwdVolChart}
                    layout={{
                      title: { text: 'Forward Vol Strip' },
                      yaxis: { title: { text: 'Forward Vol (%)' } },
                    }}
                  />
                )}
                {active === 'arb' && (
                  <div>
                    {arbCount === 0 ? (
                      <div className="text-center py-8">
                        <p className="text-positive font-medium">No calendar arbitrage violations</p>
                        <p className="text-muted text-sm mt-1">Total variance is monotonically increasing</p>
                      </div>
                    ) : (
                      <table className="w-full text-xs">
                        <thead><tr className="border-b border-border">
                          {['Tenor', 'Days', 'ATM IV', 'Fwd Vol', 'Total Var'].map(h => (
                            <th key={h} className="text-left px-3 py-2 text-muted uppercase">{h}</th>
                          ))}
                        </tr></thead>
                        <tbody>
                          {ts.filter(p => p.is_calendar_arb).map((p, i) => (
                            <tr key={i} className="bg-negative/5 border-b border-border/40">
                              <td className="px-3 py-2 font-mono text-negative">{p.tenor_label}</td>
                              <td className="px-3 py-2 font-mono">{p.days.toFixed(0)}</td>
                              <td className="px-3 py-2 font-mono">{fmtPct(p.atm_iv_pct)}</td>
                              <td className="px-3 py-2 font-mono text-negative">{p.fwd_vol_pct != null ? fmtPct(p.fwd_vol_pct) : '—'}</td>
                              <td className="px-3 py-2 font-mono">{p.total_var.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                )}
                {active === 'totalvar' && (
                  <PlotlyChart
                    data={totalVarChart}
                    layout={{
                      title: { text: 'Total Variance w(T) = σ²·T — must be non-decreasing' },
                      xaxis: { title: { text: 'Days to Expiry' } },
                      yaxis: { title: { text: 'w(T)' } },
                    }}
                  />
                )}
              </Card>
            )}
          </Tabs>
        </>
      )}
    </div>
  )
}
