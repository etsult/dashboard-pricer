import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Zap } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { api } from '@/lib/api'
import { fmt } from '@/lib/utils'

export default function VolStrategy() {
  const [currency, setCurrency] = useState('ETH')
  const [historyDays, setHistoryDays]   = useState(365)
  const [tDays, setTDays]               = useState(30)
  const [rebalFreq, setRebalFreq]       = useState(1)
  const [notional, setNotional]         = useState(100_000)
  const [result, setResult] = useState<any>(null) // eslint-disable-line @typescript-eslint/no-explicit-any

  const { mutate, isPending, error } = useMutation({
    mutationFn: () =>
      api.post('/research/backtest/dh-straddle', {
        currency, history_days: historyDays, T_days: tDays,
        rebalance_freq: rebalFreq, notional_usd: notional,
        costs: { spread_pct: 0.0005, commission_pct: 0.0003, slippage_pct: 0.0001, funding_rate_daily: 0.0003 },
      }).then(r => r.data),
    onSuccess: setResult,
  })

  const daily = result?.daily ?? []
  const pnlChart = daily.length ? [
    {
      x: daily.map((d: any) => d.date), // eslint-disable-line @typescript-eslint/no-explicit-any
      y: daily.map((d: any) => d.cumulative_pnl), // eslint-disable-line @typescript-eslint/no-explicit-any
      name: 'Net P&L', type: 'scatter' as const, mode: 'lines' as const,
      line: { color: '#3b82f6', width: 2 }, fill: 'tozeroy' as const,
    },
  ] : []

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Zap size={20} className="text-yellow-400" />
        <div>
          <h1 className="text-xl font-bold text-white">Vol Premium Strategy</h1>
          <p className="text-muted text-sm mt-1">Delta-hedged straddle backtest — sell implied, buy realized vol</p>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>Backtest Parameters</CardTitle></CardHeader>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <Select label="Currency" value={currency} onChange={e => setCurrency(e.target.value)}
            options={[{ value: 'BTC', label: 'BTC' }, { value: 'ETH', label: 'ETH' }]} />
          <Input label="History (days)" type="number" value={historyDays} onChange={e => setHistoryDays(+e.target.value)} step="30" min={90} max={1000} />
          <Input label="Option Tenor (days)" type="number" value={tDays} onChange={e => setTDays(+e.target.value)} step="7" min={7} max={90} />
          <Input label="Rebalance Freq (days)" type="number" value={rebalFreq} onChange={e => setRebalFreq(+e.target.value)} min={1} max={30} />
          <Input label="Notional (USD)" type="number" value={notional} onChange={e => setNotional(+e.target.value)} step="10000" />
        </div>
        <div className="mt-4">
          <Button onClick={() => mutate()} disabled={isPending}>
            {isPending ? 'Running backtest…' : 'Run Backtest'}
          </Button>
          {error && <p className="text-negative text-sm mt-2">Error: {(error as Error).message}</p>}
        </div>
      </Card>

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Gross Return" value={`${fmt(result.gross_performance?.total_return_pct)}%`}
              color={result.gross_performance?.total_return_pct >= 0 ? 'text-positive' : 'text-negative'} />
            <Stat label="Net Return" value={`${fmt(result.net_performance?.total_return_pct)}%`}
              color={result.net_performance?.total_return_pct >= 0 ? 'text-positive' : 'text-negative'} />
            <Stat label="Sharpe" value={fmt(result.net_performance?.sharpe_ratio)} />
            <Stat label="Total Costs" value={`$${(result.cost_summary?.total_costs ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} color="text-negative" />
          </div>
          <Card>
            <CardHeader><CardTitle>Cumulative P&L</CardTitle></CardHeader>
            <PlotlyChart
              data={pnlChart}
              layout={{ xaxis: { title: { text: 'Date' } }, yaxis: { title: { text: 'P&L (USD)' } } }}
            />
          </Card>
        </>
      )}
    </div>
  )
}
