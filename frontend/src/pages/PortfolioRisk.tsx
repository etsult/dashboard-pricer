import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Shield } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { generateBook, riskBook, type BookResponse, type RiskResponse } from '@/lib/api'
import { fmt } from '@/lib/utils'

const DEFAULT_CURVE_PTS: [number, number][] = [
  [0.25, 0.052], [0.5, 0.053], [1, 0.054],
  [2, 0.049], [3, 0.046], [5, 0.043], [10, 0.041],
]

const SHIFTS = [-200, -100, -50, 0, 50, 100, 200]

export default function PortfolioRisk() {
  const [n, setN]           = useState(500)
  const [book, setBook]     = useState<BookResponse | null>(null)
  const [risk, setRisk]     = useState<RiskResponse | null>(null)
  const [scenarios, setScenarios] = useState<{ shift: number; pv: number }[]>([])

  const genMut = useMutation({
    mutationFn: () => generateBook({ n, seed: 99, usd_weight: 0.6, add_hedges: true }),
    onSuccess: (data) => { setBook(data); setRisk(null); setScenarios([]) },
  })

  const riskMut = useMutation({
    mutationFn: async () => {
      const base = await riskBook({
        positions: book!.positions,
        curve: { points: DEFAULT_CURVE_PTS, shift_bp: 0 },
      })
      setRisk(base)

      // Run scenario analysis
      const pvs = await Promise.all(
        SHIFTS.map(shift =>
          riskBook({
            positions: book!.positions,
            curve: { points: DEFAULT_CURVE_PTS, shift_bp: shift },
          }).then(r => ({ shift, pv: r.aggregate.total_pv }))
        )
      )
      setScenarios(pvs)
    },
  })

  const scenarioChart = scenarios.length ? [{
    x: scenarios.map(s => `${s.shift > 0 ? '+' : ''}${s.shift}bp`),
    y: scenarios.map(s => s.pv),
    type: 'bar' as const,
    marker: { color: scenarios.map(s => s.pv >= 0 ? '#22c55e' : '#ef4444') },
    name: 'Portfolio PV',
  }] : []

  const dv01Chart = risk ? Object.entries(risk.aggregate.by_expiry).map(([k, v], i) => ({
    x: [k], y: [v.dv01],
    name: k, type: 'bar' as const,
    marker: { color: v.dv01 >= 0 ? '#3b82f6' : '#ef4444' },
  })) : []

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Shield size={20} className="text-red-400" />
        <div>
          <h1 className="text-xl font-bold text-white">Portfolio Risk</h1>
          <p className="text-muted text-sm mt-1">IR option book risk — scenario analysis &amp; DV01 bucketing</p>
        </div>
      </div>

      <Card>
        <CardHeader><CardTitle>Book Setup</CardTitle></CardHeader>
        <div className="flex items-end gap-4">
          <div className="w-48">
            <Input label="Positions" type="number" value={n} onChange={e => setN(+e.target.value)} step={100} min={100} max={10000} />
          </div>
          <Button onClick={() => genMut.mutate()} disabled={genMut.isPending}>
            {genMut.isPending ? 'Generating…' : 'Generate Book'}
          </Button>
          {book && (
            <Button variant="outline" onClick={() => riskMut.mutate()} disabled={riskMut.isPending}>
              {riskMut.isPending ? 'Running scenarios…' : 'Run Scenario Analysis'}
            </Button>
          )}
        </div>
        {genMut.error  && <p className="text-negative text-sm mt-2">Error: {(genMut.error  as Error).message}</p>}
        {riskMut.error && <p className="text-negative text-sm mt-2">Error: {(riskMut.error as Error).message}</p>}
      </Card>

      {book && !risk && (
        <p className="text-muted text-sm">Book generated ({book.n_positions} positions). Click "Run Scenario Analysis" to compute risk.</p>
      )}

      {risk && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Total PV ($M)" value={fmt(risk.aggregate.total_pv / 1e6)} />
            <Stat label="Total DV01 ($)" value={fmt(risk.aggregate.total_dv01)}
              color={risk.aggregate.total_dv01 < 0 ? 'text-negative' : 'text-positive'} />
            <Stat label="Gamma Up" value={fmt(risk.aggregate.total_gamma_up)} />
            <Stat label="Gamma Down" value={fmt(risk.aggregate.total_gamma_dn)} />
          </div>

          <Card>
            <CardHeader><CardTitle>Parallel Shift Scenario PV</CardTitle></CardHeader>
            <PlotlyChart
              data={scenarioChart}
              layout={{
                xaxis: { title: { text: 'Rate Shift' } },
                yaxis: { title: { text: 'Portfolio PV ($)' } },
                showlegend: false,
              }}
            />
          </Card>

          <Card>
            <CardHeader><CardTitle>DV01 by Expiry Bucket</CardTitle></CardHeader>
            <PlotlyChart
              data={dv01Chart}
              layout={{ yaxis: { title: { text: 'DV01 ($)' } }, showlegend: false, barmode: 'group' }}
            />
          </Card>

          <Card>
            <CardHeader><CardTitle>Risk by Index</CardTitle></CardHeader>
            <table className="w-full text-xs">
              <thead><tr className="border-b border-border">
                {['Index', 'PV ($)', 'DV01 ($)', 'Gamma Up', 'Gamma Dn'].map(h => (
                  <th key={h} className="text-left px-3 py-2 text-muted uppercase">{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {Object.entries(risk.aggregate.by_index).map(([idx, v]) => (
                  <tr key={idx} className="border-b border-border/40 hover:bg-panel/50">
                    <td className="px-3 py-2 font-mono text-slate-300">{idx}</td>
                    <td className="px-3 py-2 font-mono">{v.pv.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                    <td className={`px-3 py-2 font-mono ${v.dv01 < 0 ? 'text-negative' : 'text-positive'}`}>{fmt(v.dv01)}</td>
                    <td className="px-3 py-2 font-mono">{fmt(v.gamma_up)}</td>
                    <td className="px-3 py-2 font-mono">{fmt(v.gamma_dn)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  )
}
