import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { DollarSign } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs } from '@/components/ui/tabs'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { api } from '@/lib/api'
import { fmt } from '@/lib/utils'

const DEFAULT_CURVE = [
  { tenor: 0.25, rate: 0.052 }, { tenor: 0.5, rate: 0.053 },
  { tenor: 1,   rate: 0.054 }, { tenor: 2,   rate: 0.049 },
  { tenor: 3,   rate: 0.046 }, { tenor: 5,   rate: 0.043 },
  { tenor: 7,   rate: 0.042 }, { tenor: 10,  rate: 0.041 },
  { tenor: 20,  rate: 0.040 }, { tenor: 30,  rate: 0.040 },
]

export default function RatesHub() {
  const [curvePoints, setCurvePoints] = useState(DEFAULT_CURVE)
  const [curveResult, setCurveResult] = useState<{ points: { tenor: number; tenor_label: string; zero_rate_pct: number; discount_factor: number }[] } | null>(null)

  // Bond pricer state
  const [faceValue, setFaceValue]   = useState(1000)
  const [coupon, setCoupon]         = useState(4.5)
  const [ytm, setYtm]               = useState(4.0)
  const [nPeriods, setNPeriods]     = useState(10)

  const curveMut = useMutation({
    mutationFn: () => api.post('/ir/curve', { type: 'manual', points: curvePoints }).then(r => r.data),
    onSuccess: setCurveResult,
  })

  // Bond analytics (computed in browser)
  const bondCalc = () => {
    const y = ytm / 100
    const c = (coupon / 100) * faceValue / 2
    const n = nPeriods * 2  // semi-annual
    const r = y / 2
    let price = 0
    let duration = 0
    for (let t = 1; t <= n; t++) {
      const pv = c / Math.pow(1 + r, t)
      price += pv
      duration += t * pv
    }
    price += faceValue / Math.pow(1 + r, n)
    duration += n * (faceValue / Math.pow(1 + r, n))
    duration = (duration / price) / 2  // Macaulay, annualized
    const modDuration = duration / (1 + r)
    const dv01 = price * modDuration * 0.0001
    return { price, duration, modDuration, dv01 }
  }

  const bond = bondCalc()

  const pts = curveResult?.points ?? []
  const zeroChart = [{
    x: pts.map(p => p.tenor), y: pts.map(p => p.zero_rate_pct),
    name: 'Zero Rate', type: 'scatter' as const, mode: 'lines+markers' as const,
    line: { color: '#3b82f6', width: 2 },
  }]

  const dfChart = [{
    x: pts.map(p => p.tenor), y: pts.map(p => p.discount_factor),
    name: 'Discount Factor', type: 'scatter' as const, mode: 'lines' as const,
    fill: 'tozeroy' as const, line: { color: '#22c55e', width: 2 },
  }]

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <DollarSign size={20} className="text-teal-400" />
        <div>
          <h1 className="text-xl font-bold text-white">Rates Hub</h1>
          <p className="text-muted text-sm mt-1">Yield curve bootstrapping, bond analytics, duration</p>
        </div>
      </div>

      <Tabs tabs={[{ id: 'curve', label: 'Yield Curve' }, { id: 'bond', label: 'Bond Pricer' }]}>
        {active => (
          <>
            {active === 'curve' && (
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Manual Curve (Par Yields)</CardTitle>
                    <Button size="sm" onClick={() => curveMut.mutate()} disabled={curveMut.isPending}>
                      {curveMut.isPending ? 'Bootstrapping…' : 'Bootstrap Curve'}
                    </Button>
                  </CardHeader>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    {curvePoints.map((pt, i) => (
                      <div key={i} className="flex flex-col gap-1">
                        <label className="text-xs text-muted">{pt.tenor}Y rate</label>
                        <input
                          type="number" step="0.001" value={pt.rate}
                          onChange={e => setCurvePoints(pts => pts.map((p, idx) => idx === i ? { ...p, rate: +e.target.value } : p))}
                          className="h-8 rounded border border-border bg-panel px-2 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-accent"
                        />
                      </div>
                    ))}
                  </div>
                  {curveMut.error && <p className="text-negative text-sm mt-2">Error: {(curveMut.error as Error).message}</p>}
                </Card>

                {curveResult && (
                  <>
                    <Tabs tabs={[{ id: 'zero', label: 'Zero Rates' }, { id: 'df', label: 'Discount Factors' }]}>
                      {sub => (
                        <Card>
                          {sub === 'zero' && (
                            <PlotlyChart data={zeroChart} layout={{ xaxis: { title: { text: 'Tenor (yr)' } }, yaxis: { title: { text: 'Zero Rate (%)' } } }} />
                          )}
                          {sub === 'df' && (
                            <PlotlyChart data={dfChart} layout={{ xaxis: { title: { text: 'Tenor (yr)' } }, yaxis: { title: { text: 'Discount Factor' } } }} />
                          )}
                        </Card>
                      )}
                    </Tabs>

                    <Card>
                      <CardHeader><CardTitle>Curve Data</CardTitle></CardHeader>
                      <table className="w-full text-xs">
                        <thead><tr className="border-b border-border">
                          {['Tenor', 'Zero Rate', 'Discount Factor'].map(h => (
                            <th key={h} className="text-left px-3 py-2 text-muted uppercase">{h}</th>
                          ))}
                        </tr></thead>
                        <tbody>
                          {pts.map((p, i) => (
                            <tr key={i} className="border-b border-border/40 hover:bg-panel/50">
                              <td className="px-3 py-2 font-mono text-slate-300">{p.tenor_label}</td>
                              <td className="px-3 py-2 font-mono text-blue-400">{p.zero_rate_pct.toFixed(3)}%</td>
                              <td className="px-3 py-2 font-mono">{p.discount_factor.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </Card>
                  </>
                )}
              </div>
            )}

            {active === 'bond' && (
              <div className="space-y-4">
                <Card>
                  <CardHeader><CardTitle>Bond Parameters</CardTitle></CardHeader>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <Input label="Face Value ($)" type="number" value={faceValue} onChange={e => setFaceValue(+e.target.value)} step="100" />
                    <Input label="Coupon Rate (%)" type="number" value={coupon} onChange={e => setCoupon(+e.target.value)} step="0.25" />
                    <Input label="YTM (%)" type="number" value={ytm} onChange={e => setYtm(+e.target.value)} step="0.1" />
                    <Input label="Years to Maturity" type="number" value={nPeriods} onChange={e => setNPeriods(+e.target.value)} step="1" min={1} />
                  </div>
                  <p className="text-xs text-muted mt-2">Analytics computed in real-time using semi-annual compounding.</p>
                </Card>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <Stat label="Dirty Price" value={`$${fmt(bond.price)}`} color={bond.price > faceValue ? 'text-positive' : 'text-negative'} sub={`${fmt(bond.price / faceValue * 100)} of par`} />
                  <Stat label="Macaulay Dur" value={`${fmt(bond.duration)} yr`} />
                  <Stat label="Modified Dur" value={`${fmt(bond.modDuration)} yr`} />
                  <Stat label="DV01" value={`$${fmt(bond.dv01, 4)}`} sub="per $1 notional" />
                </div>
              </div>
            )}
          </>
        )}
      </Tabs>
    </div>
  )
}
