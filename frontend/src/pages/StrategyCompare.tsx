import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { GitCompare } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { compareStrategies, runTimingAnalysis, type TimingAnalysisResult } from '@/lib/api'
import { fmt } from '@/lib/utils'

export default function StrategyCompare() {
  const [ticker, setTicker]   = useState('SPY')
  const [start, setStart]     = useState('2018-01-01')
  const [end, setEnd]         = useState(new Date().toISOString().slice(0, 10))
  const [amount, setAmount]   = useState(10000)
  const [freq, setFreq]       = useState('monthly')
  const [cost, setCost]       = useState(0)
  const [result, setResult]       = useState<{ lump_sum: unknown; dca: unknown } | null>(null)
  const [timing, setTiming]       = useState<TimingAnalysisResult | null>(null)

  const { mutate, isPending, error } = useMutation({
    mutationFn: () => compareStrategies({
      ticker, start, end, total_amount: amount,
      dca_frequency: freq, transaction_cost_pct: cost / 100,
    }),
    onSuccess: (data) => { setResult(data); setTiming(null) },
  })

  const timingMut = useMutation({
    mutationFn: () => runTimingAnalysis({
      ticker, start, end, total_amount: amount,
      transaction_cost_pct: cost / 100, sample_every_n_days: 5,
    }),
    onSuccess: setTiming,
  })

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ls: any  = result?.lump_sum
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const dca: any = result?.dca

  const portfolioChart = result ? [
    {
      x: ls.daily.map((d: { date: string }) => d.date),
      y: ls.daily.map((d: { portfolio_value: number }) => d.portfolio_value),
      name: 'Lump Sum', type: 'scatter' as const, mode: 'lines' as const,
      line: { color: '#3b82f6', width: 2 },
    },
    {
      x: dca.daily.map((d: { date: string }) => d.date),
      y: dca.daily.map((d: { portfolio_value: number }) => d.portfolio_value),
      name: 'DCA', type: 'scatter' as const, mode: 'lines' as const,
      line: { color: '#22c55e', width: 2 },
    },
  ] : []

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <GitCompare size={20} className="text-accent" />
        <div>
          <h1 className="text-xl font-bold text-white">Strategy Compare</h1>
          <p className="text-muted text-sm">Lump Sum vs DCA equity backtest</p>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>Parameters</CardTitle></CardHeader>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <Input label="Ticker" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="SPY" />
          <Input label="Start Date" type="date" value={start} onChange={e => setStart(e.target.value)} />
          <Input label="End Date"   type="date" value={end}   onChange={e => setEnd(e.target.value)} />
          <Input label="Total Amount ($)" type="number" value={amount} onChange={e => setAmount(+e.target.value)} step="1000" />
          <Select label="DCA Frequency" value={freq} onChange={e => setFreq(e.target.value)}
            options={[
              { value: 'weekly',     label: 'Weekly'     },
              { value: 'biweekly',   label: 'Bi-weekly'  },
              { value: 'monthly',    label: 'Monthly'    },
              { value: 'quarterly',  label: 'Quarterly'  },
            ]}
          />
          <Input label="Transaction Cost (%)" type="number" value={cost} onChange={e => setCost(+e.target.value)} step="0.01" min={0} max={2} />
        </div>
        <div className="mt-4">
          <Button onClick={() => mutate()} disabled={isPending}>
            <GitCompare size={15} />
            {isPending ? 'Running backtest…' : 'Compare'}
          </Button>
          {error && <p className="text-negative text-sm mt-2">Error: {(error as Error).message}. Is yfinance data available?</p>}
        </div>
      </Card>

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="LS Final Value" value={`$${ls.performance.final_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="LS Return" value={`${fmt(ls.performance.total_return_pct)}%`}
              color={ls.performance.total_return_pct >= 0 ? 'text-positive' : 'text-negative'} />
            <Stat label="DCA Final Value" value={`$${dca.performance.final_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="DCA Return" value={`${fmt(dca.performance.total_return_pct)}%`}
              color={dca.performance.total_return_pct >= 0 ? 'text-positive' : 'text-negative'} />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="LS Sharpe"              value={fmt(ls.performance.sharpe)} />
            <Stat label="LS Max DD"              value={`${fmt(ls.performance.max_drawdown_pct)}%`} color="text-negative" />
            <Stat label="LS DD Recovery (days)"  value={ls.performance.max_dd_duration_days ?? '—'} />
            <Stat label="LS % time in DD"        value={`${fmt(ls.performance.pct_in_drawdown)}%`} color="text-negative" />
            <Stat label="DCA Sharpe"             value={fmt(dca.performance.sharpe)} />
            <Stat label="DCA Max DD"             value={`${fmt(dca.performance.max_drawdown_pct)}%`} color="text-negative" />
            <Stat label="DCA DD Recovery (days)" value={dca.performance.max_dd_duration_days ?? '—'} />
            <Stat label="DCA % time in DD"       value={`${fmt(dca.performance.pct_in_drawdown)}%`} color="text-negative" />
          </div>

          <Card>
            <CardHeader><CardTitle>Portfolio Value Over Time</CardTitle></CardHeader>
            <PlotlyChart
              data={portfolioChart}
              layout={{
                xaxis: { title: { text: 'Date' } },
                yaxis: { title: { text: 'Portfolio Value ($)' } },
              }}
            />
          </Card>

          {/* Timing analysis trigger */}
          <Card>
            <CardHeader>
              <CardTitle>Timing Risk — What If You Picked the Worst Moment?</CardTitle>
              <Button
                variant="outline"
                size="sm"
                onClick={() => timingMut.mutate()}
                disabled={timingMut.isPending}
              >
                {timingMut.isPending ? 'Analysing…' : 'Run Timing Analysis'}
              </Button>
            </CardHeader>
            <p className="text-muted text-xs">
              Simulates investing ${amount.toLocaleString()} on every possible date.
              Shows the distribution of outcomes, worst/best timing, and drawdown from worst entry.
            </p>
            {timingMut.error && (
              <p className="text-negative text-xs mt-2">{(timingMut.error as Error).message}</p>
            )}
          </Card>
        </>
      )}

      {/* ── Timing analysis results ── */}
      {timing && (() => {
        const returnScatter = [{
          x: timing.by_entry.map(e => e.date),
          y: timing.by_entry.map(e => e.final_return_pct),
          type: 'scatter' as const,
          mode: 'markers' as const,
          marker: {
            color: timing.by_entry.map(e => e.final_return_pct >= 0 ? '#22c55e' : '#ef4444'),
            size: 4, opacity: 0.7,
          },
          name: 'Final return by entry date',
          hovertemplate: '%{x}<br>Return: %{y:.1f}%<extra></extra>',
        }]

        const ddScatter = [{
          x: timing.worst_dd_path.dates,
          y: timing.worst_dd_path.drawdown_pct,
          type: 'scatter' as const,
          mode: 'lines' as const,
          fill: 'tozeroy' as const,
          fillcolor: 'rgba(239,68,68,0.15)',
          line: { color: '#ef4444', width: 1.5 },
          name: 'Drawdown from worst entry',
        }]

        const pvWorstChart = [{
          x: timing.worst_dd_path.dates,
          y: timing.worst_dd_path.portfolio_value,
          type: 'scatter' as const,
          mode: 'lines' as const,
          line: { color: '#f4a261', width: 2 },
          name: `Worst entry (${timing.worst_entry.date})`,
        }]

        return (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Worst entry return"    value={`${fmt(timing.worst_entry.final_return_pct)}%`}
                color="text-negative" sub={`Entry: ${timing.worst_entry.date}`} />
              <Stat label="Best entry return"     value={`${fmt(timing.best_entry.final_return_pct)}%`}
                color="text-positive" sub={`Entry: ${timing.best_entry.date}`} />
              <Stat label="Median return"         value={`${fmt(timing.median_return_pct)}%`} />
              <Stat label="% entries profitable"  value={`${timing.pct_entries_positive}%`}
                color={timing.pct_entries_positive >= 50 ? 'text-positive' : 'text-negative'} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              {timing.worst_entry.time_to_recover_days !== null ? (
                <Stat label="Worst entry — DD recovery"
                  value={`${timing.worst_entry.time_to_recover_days} trading days`}
                  sub={timing.worst_entry.ever_recovered ? 'Recovered' : 'Never recovered in period'}
                  color={timing.worst_entry.ever_recovered ? 'text-positive' : 'text-negative'} />
              ) : (
                <Stat label="Worst entry — DD recovery" value="Never recovered" color="text-negative" />
              )}
              <Stat label="Entry dates analysed" value={timing.n_entries.toLocaleString()} />
            </div>

            <Card>
              <CardHeader><CardTitle>Return Distribution by Entry Date</CardTitle></CardHeader>
              <p className="text-muted text-xs mb-3">
                Each dot = investing ${amount.toLocaleString()} on that date and holding until {timing.end_date}.
                Green = profitable, red = loss.
              </p>
              <PlotlyChart
                data={returnScatter}
                layout={{
                  xaxis: { title: { text: 'Entry date' } },
                  yaxis: { title: { text: 'Final return (%)' }, zeroline: true, zerolinecolor: '#6b7280' },
                  shapes: [
                    { type: 'line', x0: timing.worst_entry.date, x1: timing.worst_entry.date,
                      y0: 0, y1: 1, yref: 'paper', line: { color: '#ef4444', dash: 'dot', width: 1 } },
                    { type: 'line', x0: timing.best_entry.date,  x1: timing.best_entry.date,
                      y0: 0, y1: 1, yref: 'paper', line: { color: '#22c55e', dash: 'dot', width: 1 } },
                  ],
                }}
              />
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Portfolio Value — Worst Entry ({timing.worst_entry.date})</CardTitle>
                </CardHeader>
                <PlotlyChart
                  data={pvWorstChart}
                  layout={{ yaxis: { title: { text: 'Value ($)' } } }}
                  style={{ height: 260 }}
                />
              </Card>
              <Card>
                <CardHeader><CardTitle>Drawdown from Worst Entry</CardTitle></CardHeader>
                <PlotlyChart
                  data={ddScatter}
                  layout={{ yaxis: { title: { text: 'Drawdown (%)' }, zeroline: true, zerolinecolor: '#6b7280' } }}
                  style={{ height: 260 }}
                />
              </Card>
            </div>
          </>
        )
      })()}
    </div>
  )
}
