import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Activity } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { fmt } from '@/lib/utils'

// Abramowitz & Stegun approximation of erf (not in browser Math)
function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - (0.254829592 * t - 0.284496736 * t ** 2 + 1.421413741 * t ** 3
    - 1.453152027 * t ** 4 + 1.061405429 * t ** 5) * Math.exp(-x * x)
  return Math.sign(x) * y
}

function simulateDeltaHedge(
  S0: number, K: number, T: number, sigma: number, r: number, freq: number, n: number = 252
) {
  // Monte Carlo path + hedging simulation (simplified Black-Scholes)
  const dt = T / n
  const sqrtDt = Math.sqrt(dt)
  const rebalDays = Math.round(freq)

  // Generate price path
  const path: number[] = [S0]
  for (let i = 1; i <= n; i++) {
    const z = (Math.random() - 0.5) * 2 * 1.7  // approx normal
    path.push(path[i - 1] * Math.exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrtDt * z))
  }

  // Black-Scholes delta
  function bsDelta(S: number, tRemaining: number): number {
    if (tRemaining <= 0) return S > K ? 1 : 0
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * tRemaining) / (sigma * Math.sqrt(tRemaining))
    return 0.5 * (1 + erf(d1 / Math.sqrt(2)))
  }

  // Hedging P&L
  const pnlHedged: number[] = [0]
  const pnlNaked: number[] = [0]
  let cashHedged = 0
  let shares = 0
  let lastRebal = 0

  for (let i = 1; i <= n; i++) {
    const tRem = T - i * dt
    // Naked: just hold the option value change
    pnlNaked.push(pnlNaked[i - 1] + (path[i] - path[i - 1]) * (i === n ? (path[i] > K ? 1 : 0) : bsDelta(path[i - 1], tRem + dt)))
    // Hedged
    if (i - lastRebal >= rebalDays || i === n) {
      const newDelta = bsDelta(path[i], tRem)
      cashHedged -= (newDelta - shares) * path[i]
      shares = newDelta
      lastRebal = i
    }
    const optionPnl = path[i] > K ? path[i] - K : 0
    pnlHedged.push(cashHedged + shares * path[i] + (i === n ? optionPnl : 0))
  }

  return { path, pnlHedged, pnlNaked, n }
}

export default function DeltaHedge() {
  const [spot, setSpot]   = useState(100)
  const [strike, setStrike] = useState(100)
  const [T, setT]         = useState(0.25)
  const [sigma, setSigma] = useState(0.2)
  const [rate, setRate]   = useState(0.05)
  const [freq, setFreq]   = useState(5)
  const [result, setResult] = useState<ReturnType<typeof simulateDeltaHedge> | null>(null)

  const { mutate, isPending } = useMutation({
    mutationFn: async () => simulateDeltaHedge(spot, strike, T, sigma, rate, freq),
    onSuccess: setResult,
  })

  const days = result ? Array.from({ length: result.n + 1 }, (_, i) => i) : []

  const priceChart = result ? [{
    x: days, y: result.path,
    name: 'Spot Price', type: 'scatter' as const, mode: 'lines' as const,
    line: { color: '#3b82f6', width: 1.5 },
  }] : []

  const pnlChart = result ? [
    {
      x: days, y: result.pnlHedged,
      name: 'Delta Hedged', type: 'scatter' as const, mode: 'lines' as const,
      line: { color: '#22c55e', width: 2 },
    },
    {
      x: days, y: result.pnlNaked,
      name: 'Naked Long', type: 'scatter' as const, mode: 'lines' as const,
      line: { color: '#a78bfa', width: 2, dash: 'dot' },
    },
  ] : []

  const finalHedged = result?.pnlHedged[result.n] ?? 0
  const finalNaked  = result?.pnlNaked[result.n] ?? 0

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Activity size={20} className="text-pink-400" />
        <div>
          <h1 className="text-xl font-bold text-white">Delta Hedge Simulator</h1>
          <p className="text-muted text-sm mt-1">Simulate delta hedging a long call option — gamma vs theta P&L</p>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>Parameters</CardTitle></CardHeader>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <Input label="Spot (S₀)" type="number" value={spot}   onChange={e => setSpot(+e.target.value)} step="5" />
          <Input label="Strike (K)" type="number" value={strike} onChange={e => setStrike(+e.target.value)} step="5" />
          <Input label="Time to Expiry (yr)" type="number" value={T} onChange={e => setT(+e.target.value)} step="0.0833" min={0.01} />
          <Input label="Volatility σ" type="number" value={sigma} onChange={e => setSigma(+e.target.value)} step="0.01" min={0.01} />
          <Input label="Risk-Free Rate" type="number" value={rate} onChange={e => setRate(+e.target.value)} step="0.01" />
          <Select label="Rehedge Frequency" value={String(freq)} onChange={e => setFreq(+e.target.value)}
            options={[
              { value: '1',  label: 'Daily'   },
              { value: '5',  label: 'Weekly'  },
              { value: '21', label: 'Monthly' },
            ]}
          />
        </div>
        <div className="mt-4">
          <Button onClick={() => mutate()} disabled={isPending}>
            {isPending ? 'Simulating…' : 'Run Simulation'}
          </Button>
          <p className="text-xs text-muted mt-2">Runs one Monte Carlo path in the browser. Each run gives a different path.</p>
        </div>
      </Card>

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Hedged P&L" value={fmt(finalHedged)} color={finalHedged >= 0 ? 'text-positive' : 'text-negative'} />
            <Stat label="Naked P&L"  value={fmt(finalNaked)}  color={finalNaked  >= 0 ? 'text-positive' : 'text-negative'} />
            <Stat label="Final Spot" value={fmt(result.path[result.n])} />
            <Stat label="Rehedge Days" value={freq} />
          </div>
          <Card>
            <CardHeader><CardTitle>Simulated Price Path</CardTitle></CardHeader>
            <PlotlyChart data={priceChart} layout={{ yaxis: { title: { text: 'Spot Price' } }, xaxis: { title: { text: 'Trading Day' } } }} />
          </Card>
          <Card>
            <CardHeader><CardTitle>Cumulative P&L — Delta Hedged vs Naked Long</CardTitle></CardHeader>
            <PlotlyChart data={pnlChart} layout={{ yaxis: { title: { text: 'P&L' } }, xaxis: { title: { text: 'Trading Day' } } }} />
          </Card>
        </>
      )}
    </div>
  )
}
