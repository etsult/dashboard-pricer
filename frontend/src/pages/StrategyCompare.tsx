import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { GitCompare } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { compareStrategies } from '@/lib/api'
import { fmt } from '@/lib/utils'

export default function StrategyCompare() {
  const [ticker, setTicker]   = useState('SPY')
  const [start, setStart]     = useState('2018-01-01')
  const [end, setEnd]         = useState(new Date().toISOString().slice(0, 10))
  const [amount, setAmount]   = useState(10000)
  const [freq, setFreq]       = useState('monthly')
  const [cost, setCost]       = useState(0)
  const [result, setResult]   = useState<{ lump_sum: unknown; dca: unknown } | null>(null)

  const { mutate, isPending, error } = useMutation({
    mutationFn: () => compareStrategies({
      ticker, start, end, total_amount: amount,
      dca_frequency: freq, transaction_cost_pct: cost / 100,
    }),
    onSuccess: setResult,
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
            <Stat label="LS Sharpe"    value={fmt(ls.performance.sharpe_ratio)} />
            <Stat label="LS Max DD"    value={`${fmt(ls.performance.max_drawdown_pct)}%`} color="text-negative" />
            <Stat label="DCA Sharpe"   value={fmt(dca.performance.sharpe_ratio)} />
            <Stat label="DCA Max DD"   value={`${fmt(dca.performance.max_drawdown_pct)}%`} color="text-negative" />
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
        </>
      )}
    </div>
  )
}
