import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { BookOpen, BarChart2 } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { generateBook, riskBook, type BookResponse, type RiskResponse } from '@/lib/api'
import { fmt } from '@/lib/utils'

const DEFAULT_CURVE = [
  { tenor: 0.25, rate: 0.052 }, { tenor: 0.5, rate: 0.053 },
  { tenor: 1,   rate: 0.054 }, { tenor: 2,   rate: 0.049 },
  { tenor: 3,   rate: 0.046 }, { tenor: 5,   rate: 0.043 },
  { tenor: 10,  rate: 0.041 },
]

export default function BookGenerator() {
  const [n, setN] = useState(1000)
  const [seed, setSeed] = useState(42)
  const [usdWeight, setUsdWeight] = useState(0.6)
  const [book, setBook] = useState<BookResponse | null>(null)
  const [risk, setRisk] = useState<RiskResponse | null>(null)

  const genMut = useMutation({
    mutationFn: () => generateBook({ n, seed, usd_weight: usdWeight, add_hedges: true }),
    onSuccess: (data) => { setBook(data); setRisk(null) },
  })

  const riskMut = useMutation({
    mutationFn: () => riskBook({
      positions: book!.positions,
      curve: { points: DEFAULT_CURVE.map(p => [p.tenor, p.rate]), shift_bp: 0 },
    }),
    onSuccess: setRisk,
  })

  // Instrument mix chart
  const instrCounts = book ? Object.entries(
    book.positions.reduce<Record<string, number>>((acc, p) => {
      acc[p.instrument] = (acc[p.instrument] ?? 0) + 1
      return acc
    }, {})
  ) : []

  const pieChart = instrCounts.length ? [{
    labels: instrCounts.map(([k]) => k),
    values: instrCounts.map(([, v]) => v),
    type: 'pie' as const,
    hole: 0.4,
    marker: { colors: ['#3b82f6', '#a78bfa', '#22c55e', '#f59e0b'] },
    textinfo: 'label+percent' as const,
  }] : []

  const dv01Chart = risk ? Object.entries(risk.aggregate.by_index).map(([key, val], i) => ({
    x: [key], y: [val.dv01],
    name: key, type: 'bar' as const,
    marker: { color: ['#3b82f6','#a78bfa','#22c55e','#f59e0b','#ef4444','#06b6d4'][i % 6] },
  })) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">Book Generator</h1>
        <p className="text-muted text-sm mt-1">Synthesize large IR option books — up to 400k positions</p>
      </div>

      <Card>
        <CardHeader><CardTitle>Book Parameters</CardTitle></CardHeader>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Input label="Positions" type="number" value={n} onChange={e => setN(+e.target.value)} min={100} max={500000} step={100} />
          <Input label="Random Seed" type="number" value={seed} onChange={e => setSeed(+e.target.value)} />
          <Input label="USD Weight (0-1)" type="number" value={usdWeight} onChange={e => setUsdWeight(+e.target.value)} step="0.05" min={0} max={1} />
        </div>
        <div className="flex gap-3 mt-4">
          <Button onClick={() => genMut.mutate()} disabled={genMut.isPending}>
            <BookOpen size={15} />
            {genMut.isPending ? 'Generating…' : 'Generate Book'}
          </Button>
          {book && (
            <Button variant="outline" onClick={() => riskMut.mutate()} disabled={riskMut.isPending}>
              <BarChart2 size={15} />
              {riskMut.isPending ? 'Computing Risk…' : 'Compute Risk'}
            </Button>
          )}
        </div>
        {genMut.error && <p className="text-negative text-sm mt-2">Error: {(genMut.error as Error).message}</p>}
      </Card>

      {book && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Positions" value={book.n_positions.toLocaleString()} />
            <Stat label="Book ID" value={book.book_id.slice(0, 8) + '…'} />
          </div>
          <Card>
            <CardHeader><CardTitle>Instrument Mix</CardTitle></CardHeader>
            <PlotlyChart data={pieChart} layout={{ title: { text: '' }, showlegend: false }} style={{ height: 300 }} />
          </Card>
        </>
      )}

      {risk && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Total PV ($M)" value={fmt(risk.aggregate.total_pv / 1e6)} />
            <Stat label="Total DV01" value={fmt(risk.aggregate.total_dv01)} color={risk.aggregate.total_dv01 < 0 ? 'text-negative' : 'text-positive'} />
            <Stat label="Gamma Up" value={fmt(risk.aggregate.total_gamma_up)} />
            <Stat label="Gamma Down" value={fmt(risk.aggregate.total_gamma_dn)} />
          </div>
          <Card>
            <CardHeader><CardTitle>DV01 by Index</CardTitle></CardHeader>
            <PlotlyChart
              data={dv01Chart}
              layout={{ title: { text: '' }, yaxis: { title: { text: 'DV01 ($)' } }, barmode: 'group' }}
            />
          </Card>

          <Card>
            <CardHeader><CardTitle>Risk by Expiry Bucket</CardTitle></CardHeader>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead><tr className="border-b border-border">
                  {['Bucket', 'PV ($)', 'DV01 ($)', 'Gamma Up', 'Gamma Dn'].map(h => (
                    <th key={h} className="text-left px-3 py-2 text-muted uppercase">{h}</th>
                  ))}
                </tr></thead>
                <tbody>
                  {Object.entries(risk.aggregate.by_expiry).map(([bucket, v]) => (
                    <tr key={bucket} className="border-b border-border/40 hover:bg-panel/50">
                      <td className="px-3 py-2 font-mono text-slate-300">{bucket}</td>
                      <td className="px-3 py-2 font-mono">{v.pv.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td className={`px-3 py-2 font-mono ${v.dv01 < 0 ? 'text-negative' : 'text-positive'}`}>{fmt(v.dv01)}</td>
                      <td className="px-3 py-2 font-mono">{fmt(v.gamma_up)}</td>
                      <td className="px-3 py-2 font-mono">{fmt(v.gamma_dn)}</td>
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
