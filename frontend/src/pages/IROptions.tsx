import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Calculator } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Tabs } from '@/components/ui/tabs'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { priceCapFloor, priceSwaption, type CapFloorResponse, type SwaptionResponse } from '@/lib/api'

// Simple flat yield curve defaults
const DEFAULT_CURVE = [
  { tenor: 0.25, rate: 0.052 }, { tenor: 0.5, rate: 0.053 },
  { tenor: 1,   rate: 0.054 }, { tenor: 2,   rate: 0.049 },
  { tenor: 3,   rate: 0.046 }, { tenor: 5,   rate: 0.043 },
  { tenor: 7,   rate: 0.042 }, { tenor: 10,  rate: 0.041 },
]

export default function IROptions() {
  const [instrType, setInstrType] = useState<'cap' | 'floor' | 'payer' | 'receiver'>('cap')
  const [notional, setNotional] = useState(10_000_000)
  const [maturity, setMaturity] = useState(5)
  const [expiry, setExpiry] = useState(1)
  const [swapTenor, setSwapTenor] = useState(5)
  const [freq, setFreq] = useState(0.25)
  const [volType, setVolType] = useState<'normal' | 'lognormal'>('normal')
  const [sigma, setSigma] = useState(100)   // bps for normal, % for lognormal
  const [strike, setStrike] = useState(0.04)

  const [capResult,  setCapResult]  = useState<CapFloorResponse | null>(null)
  const [swapResult, setSwapResult] = useState<SwaptionResponse | null>(null)

  const isSwaption = instrType === 'payer' || instrType === 'receiver'
  const curve = { type: 'manual' as const, points: DEFAULT_CURVE }
  const sigmaDecimal = volType === 'normal' ? sigma / 10000 : sigma / 100

  const { mutate, isPending, error } = useMutation<CapFloorResponse | SwaptionResponse>({
    mutationFn: () => isSwaption
      ? priceSwaption({ curve, swaption_type: instrType as 'payer' | 'receiver', notional, expiry, swap_tenor: swapTenor, freq, vol_type: volType, sigma: sigmaDecimal, strike })
      : priceCapFloor({ curve, instrument_type: instrType as 'cap' | 'floor', notional, maturity, freq, vol_type: volType, sigma: sigmaDecimal, strike }),
    onSuccess: (res) => {
      if (isSwaption) setSwapResult(res as SwaptionResponse)
      else setCapResult(res as CapFloorResponse)
    },
  })

  const sensitivityChart = (data: { x: number; price: number }[], label: string, color: string) => [{
    x: data.map(p => p.x), y: data.map(p => p.price),
    name: label, type: 'scatter' as const, mode: 'lines' as const,
    line: { color, width: 2 },
  }]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">IR Options Pricer</h1>
        <p className="text-muted text-sm mt-1">Caps, floors and swaptions — Bachelier (normal) or Black (lognormal)</p>
      </div>

      <Card>
        <CardHeader><CardTitle>Instrument Parameters</CardTitle></CardHeader>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Select label="Instrument" value={instrType}
            onChange={e => setInstrType(e.target.value as typeof instrType)}
            options={[
              { value: 'cap',      label: 'Cap'            },
              { value: 'floor',    label: 'Floor'          },
              { value: 'payer',    label: 'Payer Swaption' },
              { value: 'receiver', label: 'Receiver Swaption' },
            ]}
          />
          <Select label="Vol Type" value={volType}
            onChange={e => setVolType(e.target.value as 'normal' | 'lognormal')}
            options={[{ value: 'normal', label: 'Normal (bps)' }, { value: 'lognormal', label: 'Lognormal (%)' }]}
          />
          <Input label="Notional" type="number" value={notional} onChange={e => setNotional(+e.target.value)} step="1000000" />
          <Input label={volType === 'normal' ? 'Vol σ (bps)' : 'Vol σ (%)'} type="number" value={sigma} onChange={e => setSigma(+e.target.value)} step="5" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          <Input label="Strike (decimal)" type="number" value={strike} onChange={e => setStrike(+e.target.value)} step="0.001" />
          {isSwaption ? (
            <>
              <Input label="Option Expiry (yr)" type="number" value={expiry} onChange={e => setExpiry(+e.target.value)} step="0.5" />
              <Input label="Swap Tenor (yr)" type="number" value={swapTenor} onChange={e => setSwapTenor(+e.target.value)} step="1" />
            </>
          ) : (
            <Input label="Maturity (yr)" type="number" value={maturity} onChange={e => setMaturity(+e.target.value)} step="1" />
          )}
          <Select label="Payment Freq" value={String(freq)}
            onChange={e => setFreq(+e.target.value)}
            options={[{ value: '0.25', label: 'Quarterly' }, { value: '0.5', label: 'Semi-annual' }, { value: '1', label: 'Annual' }]}
          />
        </div>
        <p className="text-xs text-muted mt-3">Using hardcoded yield curve (approx SOFR). Connect FRED API key for live data.</p>

        <div className="mt-4">
          <Button onClick={() => mutate()} disabled={isPending}>
            <Calculator size={15} />
            {isPending ? 'Pricing…' : 'Price'}
          </Button>
          {error && <p className="text-negative text-sm mt-2">Error: {(error as Error).message}</p>}
        </div>
      </Card>

      {capResult && !isSwaption && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="PV ($)" value={`$${capResult.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="PV (bps)" value={capResult.price_bps.toFixed(1)} />
            <Stat label="Strike" value={`${(capResult.strike_pct).toFixed(3)}%`} />
            <Stat label="Caplets" value={capResult.caplet_details?.length ?? '—'} />
          </div>
          <Tabs tabs={[{ id: 'strike', label: 'Strike Sensitivity' }, { id: 'vol', label: 'Vol Sensitivity' }]}>
            {active => (
              <Card>
                {active === 'strike' && (
                  <PlotlyChart
                    data={sensitivityChart(capResult.sensitivity_strike, 'PV vs Strike', '#3b82f6')}
                    layout={{ title: { text: 'Price vs Strike' }, xaxis: { title: { text: 'Strike' } }, yaxis: { title: { text: 'PV ($)' } } }}
                  />
                )}
                {active === 'vol' && (
                  <PlotlyChart
                    data={sensitivityChart(capResult.sensitivity_vol, 'PV vs Vol', '#a78bfa')}
                    layout={{ title: { text: 'Price vs Volatility' }, xaxis: { title: { text: 'Vol' } }, yaxis: { title: { text: 'PV ($)' } } }}
                  />
                )}
              </Card>
            )}
          </Tabs>
        </>
      )}

      {swapResult && isSwaption && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="PV ($)" value={`$${swapResult.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="PV (bps)" value={swapResult.price_bps.toFixed(1)} />
            <Stat label="Par Swap Rate" value={`${swapResult.par_swap_rate_pct.toFixed(3)}%`} />
            <Stat label="Moneyness" value={swapResult.moneyness_label} sub={`${swapResult.moneyness_bps.toFixed(0)} bps`} />
          </div>
          <Tabs tabs={[{ id: 'strike', label: 'Strike Sensitivity' }, { id: 'vol', label: 'Vol Sensitivity' }]}>
            {active => (
              <Card>
                {active === 'strike' && (
                  <PlotlyChart
                    data={sensitivityChart(swapResult.sensitivity_strike, 'PV vs Strike', '#3b82f6')}
                    layout={{ title: { text: 'Price vs Strike' }, xaxis: { title: { text: 'Strike' } }, yaxis: { title: { text: 'PV ($)' } } }}
                  />
                )}
                {active === 'vol' && (
                  <PlotlyChart
                    data={sensitivityChart(swapResult.sensitivity_vol, 'PV vs Vol', '#a78bfa')}
                    layout={{ title: { text: 'Price vs Volatility' }, xaxis: { title: { text: 'Vol' } }, yaxis: { title: { text: 'PV ($)' } } }}
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
