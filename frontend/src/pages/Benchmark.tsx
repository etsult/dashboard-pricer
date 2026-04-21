import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Gauge } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { priceCapFloor, priceSwaption } from '@/lib/api'
import { fmt } from '@/lib/utils'

const CURVE = { type: 'manual' as const, points: [
  { tenor: 0.25, rate: 0.052 }, { tenor: 1, rate: 0.054 },
  { tenor: 2, rate: 0.049 }, { tenor: 5, rate: 0.043 }, { tenor: 10, rate: 0.041 },
] }

const STRIKES = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]

export default function Benchmark() {
  const [notional, setNotional] = useState(10_000_000)
  const [results, setResults] = useState<{
    strike: number; capPV: number; floorPV: number; payerPV: number; receiverPV: number
  }[] | null>(null)
  const [timing, setTiming] = useState<number | null>(null)

  const { mutate, isPending } = useMutation({
    mutationFn: async () => {
      const start = performance.now()
      const rows = await Promise.all(
        STRIKES.map(async (strike) => {
          const [cap, floor, payer, receiver] = await Promise.all([
            priceCapFloor({ curve: CURVE, instrument_type: 'cap',   notional, maturity: 5, freq: 0.25, vol_type: 'normal', sigma: 0.01, strike }),
            priceCapFloor({ curve: CURVE, instrument_type: 'floor', notional, maturity: 5, freq: 0.25, vol_type: 'normal', sigma: 0.01, strike }),
            priceSwaption({ curve: CURVE, swaption_type: 'payer',   notional, expiry: 1, swap_tenor: 5, freq: 0.5, vol_type: 'normal', sigma: 0.01, strike }),
            priceSwaption({ curve: CURVE, swaption_type: 'receiver',notional, expiry: 1, swap_tenor: 5, freq: 0.5, vol_type: 'normal', sigma: 0.01, strike }),
          ])
          return { strike, capPV: cap.price, floorPV: floor.price, payerPV: payer.price, receiverPV: receiver.price }
        })
      )
      return { rows, elapsed: performance.now() - start }
    },
    onSuccess: ({ rows, elapsed }) => { setResults(rows); setTiming(elapsed) },
  })

  const chartData: object[] = results ? [
    { x: results.map(r => r.strike * 100), y: results.map(r => r.capPV),      name: 'Cap',            type: 'scatter', mode: 'lines+markers', line: { color: '#3b82f6' } },
    { x: results.map(r => r.strike * 100), y: results.map(r => r.floorPV),    name: 'Floor',          type: 'scatter', mode: 'lines+markers', line: { color: '#a78bfa' } },
    { x: results.map(r => r.strike * 100), y: results.map(r => r.payerPV),    name: 'Payer Swaption', type: 'scatter', mode: 'lines+markers', line: { color: '#22c55e', dash: 'dot' } },
    { x: results.map(r => r.strike * 100), y: results.map(r => r.receiverPV), name: 'Rcvr Swaption',  type: 'scatter', mode: 'lines+markers', line: { color: '#f59e0b', dash: 'dot' } },
  ] : []

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Gauge size={20} className="text-amber-400" />
        <div>
          <h1 className="text-xl font-bold text-white">Pricer Benchmark</h1>
          <p className="text-muted text-sm mt-1">Price caps, floors &amp; swaptions across strikes — verify put-call parity</p>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>Benchmark Setup</CardTitle></CardHeader>
        <div className="flex items-end gap-4">
          <div className="w-56">
            <Input label="Notional ($)" type="number" value={notional} onChange={e => setNotional(+e.target.value)} step={1000000} />
          </div>
          <Button onClick={() => mutate()} disabled={isPending}>
            {isPending ? `Pricing ${STRIKES.length * 4} instruments…` : 'Run Benchmark'}
          </Button>
        </div>
        <p className="text-xs text-muted mt-2">Prices {STRIKES.length} strikes × 4 instruments ({STRIKES.length * 4} requests) in parallel.</p>
      </Card>

      {timing !== null && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <Stat label="Total Time" value={`${fmt(timing / 1000, 2)}s`} sub={`${STRIKES.length * 4} parallel requests`} />
          <Stat label="Avg per Request" value={`${fmt(timing / (STRIKES.length * 4), 0)}ms`} />
          <Stat label="Instruments" value={STRIKES.length * 4} />
        </div>
      )}

      {results && (
        <>
          <Card>
            <CardHeader><CardTitle>PV vs Strike — Caps, Floors &amp; Swaptions</CardTitle></CardHeader>
            <PlotlyChart
              data={chartData}
              layout={{
                xaxis: { title: { text: 'Strike (%)' } },
                yaxis: { title: { text: 'PV ($)' } },
              }}
            />
          </Card>

          <Card>
            <CardHeader><CardTitle>Results Table</CardTitle></CardHeader>
            <table className="w-full text-xs">
              <thead><tr className="border-b border-border">
                {['Strike', 'Cap ($)', 'Floor ($)', 'Payer Swap ($)', 'Rcvr Swap ($)', 'Put-Call Parity'].map(h => (
                  <th key={h} className="text-left px-3 py-2 text-muted uppercase">{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {results.map((r, i) => {
                  const parity = Math.abs(r.capPV - r.floorPV - (r.payerPV - r.receiverPV))
                  return (
                    <tr key={i} className="border-b border-border/40 hover:bg-panel/50">
                      <td className="px-3 py-2 font-mono text-slate-300">{(r.strike * 100).toFixed(2)}%</td>
                      <td className="px-3 py-2 font-mono text-blue-400">{r.capPV.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className="px-3 py-2 font-mono text-purple-400">{r.floorPV.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className="px-3 py-2 font-mono text-green-400">{r.payerPV.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className="px-3 py-2 font-mono text-yellow-400">{r.receiverPV.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className={`px-3 py-2 font-mono ${parity < 1 ? 'text-positive' : 'text-warning'}`}>
                        {parity < 1 ? '✓ OK' : `Δ$${fmt(parity, 0)}`}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  )
}
