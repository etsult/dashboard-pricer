import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Plus, Trash2, Calculator } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Tabs } from '@/components/ui/tabs'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { priceStrategy, type Leg, type StrategyResponse } from '@/lib/api'
import { fmt } from '@/lib/utils'

const defaultLeg = (): Leg => ({
  option_type: 'call', strike: 100, qty: 1, sigma: 0.2,
  expiry: new Date(Date.now() + 30 * 86400_000).toISOString().slice(0, 10),
})

export default function Strategies() {
  const today = new Date().toISOString().slice(0, 10)
  const [model, setModel] = useState<'Black-76' | 'Black-Scholes' | 'Bachelier'>('Black-Scholes')
  const [forward, setForward] = useState(100)
  const [rate, setRate] = useState(0.05)
  const [divYield, setDivYield] = useState(0)
  const [legs, setLegs] = useState<Leg[]>([defaultLeg()])
  const [result, setResult] = useState<StrategyResponse | null>(null)

  const { mutate, isPending, error } = useMutation({
    mutationFn: () => priceStrategy({
      model, forward, rate, dividend_yield: divYield,
      valuation_date: today, legs, forward_range_points: 300,
    }),
    onSuccess: setResult,
  })

  const addLeg = () => setLegs(l => [...l, defaultLeg()])
  const removeLeg = (i: number) => setLegs(l => l.filter((_, idx) => idx !== i))
  const updateLeg = (i: number, patch: Partial<Leg>) =>
    setLegs(l => l.map((leg, idx) => idx === i ? { ...leg, ...patch } : leg))

  const payoffChart = result ? [
    { x: result.payoff.forward_range, y: result.payoff.payoff_today,  name: 'P&L Today',     line: { color: '#3b82f6', width: 2 } },
    { x: result.payoff.forward_range, y: result.payoff.payoff_expiry, name: 'Payoff Expiry',  line: { color: '#22c55e', width: 2, dash: 'dot' } },
  ] : []

  const deltaChart = result ? [
    { x: result.greeks_vs_forward.forward_range, y: result.greeks_vs_forward.delta, name: 'Delta', line: { color: '#a78bfa' } },
  ] : []

  const gammaChart = result ? [
    { x: result.greeks_vs_forward.forward_range, y: result.greeks_vs_forward.gamma, name: 'Gamma', type: 'scatter' as const, fill: 'tozeroy' as const, line: { color: '#f59e0b' } },
  ] : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">Options Strategy Builder</h1>
        <p className="text-muted text-sm mt-1">Price multi-leg EQD strategies and compute Greeks</p>
      </div>

      {/* Market Inputs */}
      <Card>
        <CardHeader><CardTitle>Market Parameters</CardTitle></CardHeader>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Select label="Model" value={model} onChange={e => setModel(e.target.value as typeof model)}
            options={[{ value: 'Black-Scholes', label: 'Black-Scholes' }, { value: 'Black-76', label: 'Black-76' }, { value: 'Bachelier', label: 'Bachelier' }]} />
          <Input label="Forward / Spot" type="number" value={forward} onChange={e => setForward(+e.target.value)} step="1" />
          <Input label="Rate (decimal)" type="number" value={rate} onChange={e => setRate(+e.target.value)} step="0.001" />
          <Input label="Div Yield" type="number" value={divYield} onChange={e => setDivYield(+e.target.value)} step="0.001" />
        </div>
      </Card>

      {/* Legs */}
      <Card>
        <CardHeader>
          <CardTitle>Option Legs</CardTitle>
          <Button size="sm" variant="outline" onClick={addLeg}><Plus size={14} /> Add Leg</Button>
        </CardHeader>
        <div className="space-y-3">
          {legs.map((leg, i) => (
            <div key={i} className="grid grid-cols-2 md:grid-cols-6 gap-3 p-3 bg-panel rounded-lg border border-border items-end">
              <Select label="Type" value={leg.option_type}
                onChange={e => updateLeg(i, { option_type: e.target.value as 'call' | 'put' })}
                options={[{ value: 'call', label: 'Call' }, { value: 'put', label: 'Put' }]} />
              <Input label="Strike" type="number" value={leg.strike} onChange={e => updateLeg(i, { strike: +e.target.value })} step="1" />
              <Input label="Quantity" type="number" value={leg.qty} onChange={e => updateLeg(i, { qty: +e.target.value })} step="1" />
              <Input label="Sigma (σ)" type="number" value={leg.sigma} onChange={e => updateLeg(i, { sigma: +e.target.value })} step="0.01" min="0.001" />
              <Input label="Expiry" type="date" value={leg.expiry} onChange={e => updateLeg(i, { expiry: e.target.value })} />
              <Button size="icon" variant="danger" onClick={() => removeLeg(i)} disabled={legs.length === 1}>
                <Trash2 size={14} />
              </Button>
            </div>
          ))}
        </div>

        <div className="mt-4 flex gap-3">
          <Button onClick={() => mutate()} disabled={isPending} className="gap-2">
            <Calculator size={15} />
            {isPending ? 'Pricing…' : 'Price Strategy'}
          </Button>
        </div>

        {error && <p className="text-negative text-sm mt-3">Error: {(error as Error).message}</p>}
      </Card>

      {/* Greeks Summary */}
      {result && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          <Stat label="Price" value={fmt(result.greeks.price)} />
          <Stat label="Delta" value={fmt(result.greeks.delta)} />
          <Stat label="Gamma" value={fmt(result.greeks.gamma, 4)} />
          <Stat label="Vega"  value={fmt(result.greeks.vega)} />
          <Stat label="Theta" value={fmt(result.greeks.theta)} color={result.greeks.theta < 0 ? 'text-negative' : 'text-positive'} />
          <Stat label="Rho"   value={fmt(result.greeks.rho)} />
        </div>
      )}

      {/* Charts */}
      {result && (
        <Tabs
          tabs={[
            { id: 'payoff', label: 'Payoff' },
            { id: 'delta',  label: 'Delta' },
            { id: 'gamma',  label: 'Gamma' },
          ]}
        >
          {active => (
            <Card>
              {active === 'payoff' && (
                <PlotlyChart
                  data={payoffChart}
                  layout={{ title: { text: 'P&L vs Forward' }, xaxis: { title: { text: 'Forward' } }, yaxis: { title: { text: 'P&L' } } }}
                />
              )}
              {active === 'delta' && (
                <PlotlyChart
                  data={deltaChart}
                  layout={{ title: { text: 'Delta vs Forward' }, xaxis: { title: { text: 'Forward' } }, yaxis: { title: { text: 'Delta' } } }}
                />
              )}
              {active === 'gamma' && (
                <PlotlyChart
                  data={gammaChart}
                  layout={{ title: { text: 'Gamma vs Forward' }, xaxis: { title: { text: 'Forward' } }, yaxis: { title: { text: 'Gamma' } } }}
                />
              )}
            </Card>
          )}
        </Tabs>
      )}
    </div>
  )
}
