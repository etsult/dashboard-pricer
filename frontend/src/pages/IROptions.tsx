import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Calculator, ChevronDown, ChevronUp } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Tabs } from '@/components/ui/tabs'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { priceCapFloor, priceSwaption, type CapFloorResponse, type SwaptionResponse } from '@/lib/api'

const DEFAULT_CURVE = [
  { tenor: 0.25, rate: 0.052 }, { tenor: 0.5, rate: 0.053 },
  { tenor: 1,   rate: 0.054 }, { tenor: 2,   rate: 0.049 },
  { tenor: 3,   rate: 0.046 }, { tenor: 5,   rate: 0.043 },
  { tenor: 7,   rate: 0.042 }, { tenor: 10,  rate: 0.041 },
]

type InstrTab = 'cap' | 'floor' | 'payer' | 'receiver' | 'irs_payer' | 'irs_receiver'
type PricerModel = 'fast' | 'quantlib' | 'nn'
type DayCount = 'ACT/360' | 'ACT/365' | '30/360'

function modelBadge(model: string) {
  const colors: Record<string, string> = { fast: '#22c55e', quantlib: '#3b82f6', nn: '#a78bfa', quantlib_pending: '#f59e0b' }
  const c = colors[model] ?? '#94a3b8'
  return <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, background: `${c}22`, color: c, border: `1px solid ${c}44` }}>{model.toUpperCase()}</span>
}

export default function IROptions() {
  const [instrType, setInstrType] = useState<InstrTab>('cap')
  const [pricerModel, setPricerModel] = useState<PricerModel>('fast')

  // Common fields
  const [notional, setNotional] = useState(10_000_000)
  const [freq, setFreq] = useState(0.25)
  const [volType, setVolType] = useState<'normal' | 'lognormal'>('normal')
  const [sigma, setSigma] = useState(100)
  const [strike, setStrike] = useState(0.04)

  // Cap/Floor specific
  const [maturity, setMaturity] = useState(5)

  // Swaption specific
  const [expiry, setExpiry] = useState(1)
  const [swapTenor, setSwapTenor] = useState(5)

  // IRS specific
  const [irsType, setIrsType] = useState<'payer' | 'receiver'>('payer')
  const [irsTenor, setIrsTenor] = useState(5)
  const [fixedRate, setFixedRate] = useState(0.04)
  const [fixedFreq, setFixedFreq] = useState(0.5)
  const [floatFreq, setFloatFreq] = useState(0.25)
  const [xccy, setXccy] = useState(false)
  const [basisBps, setBasisBps] = useState(0)
  const [fxRate, setFxRate] = useState(1.0)
  const [foreignCcy, setForeignCcy] = useState('EUR')

  // Exotic / advanced params
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [startShift, setStartShift] = useState(0)
  const [dayCount, setDayCount] = useState<DayCount>('ACT/360')
  const [settlementDelay, setSettlementDelay] = useState(0)

  // Results
  const [capResult,  setCapResult]  = useState<CapFloorResponse | null>(null)
  const [swapResult, setSwapResult] = useState<SwaptionResponse | null>(null)
  const [irsResult,  setIrsResult]  = useState<IRSResponse | null>(null)

  const isSwaptionTab = instrType === 'payer' || instrType === 'receiver'
  const isIrsTab      = instrType === 'irs_payer' || instrType === 'irs_receiver'
  const curve         = { type: 'manual' as const, points: DEFAULT_CURVE }
  const sigmaDecimal  = volType === 'normal' ? sigma / 10000 : sigma / 100

  const { mutate, isPending, error } = useMutation<CapFloorResponse | SwaptionResponse>({
    mutationFn: () => isSwaption
      ? priceSwaption({ curve, swaption_type: instrType as 'payer' | 'receiver', notional, expiry, swap_tenor: swapTenor, freq, vol_type: volType, sigma: sigmaDecimal, strike })
      : priceCapFloor({ curve, instrument_type: instrType as 'cap' | 'floor', notional, maturity, freq, vol_type: volType, sigma: sigmaDecimal, strike }),
    onSuccess: (res) => {
      if (isSwaption) setSwapResult(res as unknown as SwaptionResponse)
      else setCapResult(res as unknown as CapFloorResponse)
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
        <h1 className="text-xl font-bold text-white">IR Options & Swap Pricer</h1>
        <p className="text-muted text-sm mt-1">Caps, floors, swaptions, vanilla IRS — Bachelier · Black · multi-model</p>
      </div>

      {/* Model selector */}
      <Card>
        <CardHeader><CardTitle>Pricer Model</CardTitle></CardHeader>
        <div className="flex gap-3 flex-wrap">
          {(['fast', 'quantlib', 'nn'] as PricerModel[]).map(m => (
            <button
              key={m}
              onClick={() => setPricerModel(m)}
              style={{
                padding: '6px 16px', borderRadius: 6, cursor: 'pointer',
                background: pricerModel === m ? '#3b82f620' : 'transparent',
                border: `1px solid ${pricerModel === m ? '#3b82f6' : '#2d3354'}`,
                color: pricerModel === m ? '#93c5fd' : '#94a3b8',
                fontSize: 13,
              }}
            >
              {m === 'fast' ? 'Fast (Bachelier)' : m === 'quantlib' ? 'QuantLib' : 'Neural Network'}
              {m === 'nn' && <span style={{ marginLeft: 6, fontSize: 10, color: '#f59e0b' }}>WIP</span>}
            </button>
          ))}
        </div>
        {pricerModel === 'quantlib' && (
          <p className="text-xs mt-2" style={{ color: '#94a3b8' }}>
            Uses QuantLib <code>bachelierBlackFormula</code> + exact coupon schedules + OIS discounting. Normal vol only; lognormal falls back to Black-76.
            Forward-starting caps fall back to fast engine (QL cap schedule always spot-starting).
          </p>
        )}
        {pricerModel === 'nn' && (
          <p className="text-xs mt-2" style={{ color: '#f59e0b' }}>
            NN pricer training pipeline in progress — falls back to fast Bachelier.
          </p>
        )}
      </Card>

      {/* Instrument selector */}
      <Card>
        <CardHeader><CardTitle>Instrument</CardTitle></CardHeader>
        <div className="flex gap-2 flex-wrap">
          {[
            { val: 'cap',          label: 'Cap'              },
            { val: 'floor',        label: 'Floor'            },
            { val: 'payer',        label: 'Payer Swaption'   },
            { val: 'receiver',     label: 'Receiver Swaption'},
            { val: 'irs_payer',    label: 'IRS Pay Fixed'    },
            { val: 'irs_receiver', label: 'IRS Recv Fixed'   },
          ].map(({ val, label }) => (
            <button
              key={val}
              onClick={() => setInstrType(val as InstrTab)}
              style={{
                padding: '6px 14px', borderRadius: 6, cursor: 'pointer', fontSize: 13,
                background: instrType === val ? '#6366f120' : 'transparent',
                border: `1px solid ${instrType === val ? '#6366f1' : '#2d3354'}`,
                color: instrType === val ? '#a5b4fc' : '#94a3b8',
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </Card>

      {/* Parameters */}
      <Card>
        <CardHeader><CardTitle>Parameters</CardTitle></CardHeader>

        {/* Common */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Input label="Notional" type="number" value={notional} onChange={e => setNotional(+e.target.value)} step="1000000" />
          {!isIrsTab && (
            <>
              <Select label="Vol Type" value={volType}
                onChange={e => setVolType(e.target.value as 'normal' | 'lognormal')}
                options={[{ value: 'normal', label: 'Normal (bps)' }, { value: 'lognormal', label: 'Lognormal (%)' }]}
              />
              <Input label={volType === 'normal' ? 'Vol σ (bps)' : 'Vol σ (%)'} type="number" value={sigma} onChange={e => setSigma(+e.target.value)} step="5" />
              <Input label="Strike (decimal)" type="number" value={strike} onChange={e => setStrike(+e.target.value)} step="0.001" />
            </>
          )}
        </div>

        {/* Cap/Floor params */}
        {!isSwaptionTab && !isIrsTab && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <Input label="Maturity (yr)" type="number" value={maturity} onChange={e => setMaturity(+e.target.value)} step="1" />
            <Select label="Payment Freq" value={String(freq)} onChange={e => setFreq(+e.target.value)}
              options={[{ value: '0.25', label: 'Quarterly' }, { value: '0.5', label: 'Semi-annual' }, { value: '1', label: 'Annual' }]}
            />
          </div>
        )}

        {/* Swaption params */}
        {isSwaptionTab && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <Input label="Option Expiry (yr)" type="number" value={expiry} onChange={e => setExpiry(+e.target.value)} step="0.5" />
            <Input label="Swap Tenor (yr)" type="number" value={swapTenor} onChange={e => setSwapTenor(+e.target.value)} step="1" />
            <Select label="Fixed Leg Freq" value={String(freq)} onChange={e => setFreq(+e.target.value)}
              options={[{ value: '0.25', label: 'Quarterly' }, { value: '0.5', label: 'Semi-annual' }, { value: '1', label: 'Annual' }]}
            />
          </div>
        )}

        {/* IRS params */}
        {isIrsTab && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <Input label="Swap Tenor (yr)" type="number" value={irsTenor} onChange={e => setIrsTenor(+e.target.value)} step="1" />
              <Input label="Fixed Rate (decimal)" type="number" value={fixedRate} onChange={e => setFixedRate(+e.target.value)} step="0.001" />
              <Select label="Fixed Leg Freq" value={String(fixedFreq)} onChange={e => setFixedFreq(+e.target.value)}
                options={[{ value: '0.25', label: 'Quarterly' }, { value: '0.5', label: 'Semi-annual' }, { value: '1', label: 'Annual' }]}
              />
              <Select label="Float Leg Freq" value={String(floatFreq)} onChange={e => setFloatFreq(+e.target.value)}
                options={[{ value: '0.25', label: 'Quarterly (SOFR)' }, { value: '0.5', label: 'Semi-annual' }, { value: '1', label: 'Annual' }]}
              />
            </div>

            {/* XCcy section */}
            <div className="mt-4 flex items-center gap-3">
              <label className="flex items-center gap-2 cursor-pointer text-sm text-muted">
                <input type="checkbox" checked={xccy} onChange={e => setXccy(e.target.checked)} />
                Cross-currency (XCcy)
              </label>
            </div>
            {xccy && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
                <Input label="Foreign CCY" type="text" value={foreignCcy} onChange={e => setForeignCcy(e.target.value)} />
                <Input label="FX Rate (DOM/FOR)" type="number" value={fxRate} onChange={e => setFxRate(+e.target.value)} step="0.01" />
                <Input label="Basis Spread (bps)" type="number" value={basisBps} onChange={e => setBasisBps(+e.target.value)} step="1" />
              </div>
            )}
          </>
        )}

        {/* Advanced params toggle */}
        <button
          onClick={() => setShowAdvanced(v => !v)}
          className="flex items-center gap-1 text-xs text-muted mt-4"
          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
        >
          {showAdvanced ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          Advanced / Exotic Params
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
            <Input label="Forward Start (yr)" type="number" value={startShift} onChange={e => setStartShift(+e.target.value)} step="0.5" />
            <Select label="Day Count" value={dayCount} onChange={e => setDayCount(e.target.value as DayCount)}
              options={[
                { value: 'ACT/360',  label: 'ACT/360'  },
                { value: 'ACT/365',  label: 'ACT/365F' },
                { value: '30/360',   label: '30/360'   },
              ]}
            />
            {!isIrsTab && (
              <Input label="Settlement Delay (yr)" type="number" value={settlementDelay} onChange={e => setSettlementDelay(+e.target.value)} step="0.00548" />
            )}
          </div>
        )}

        <p className="text-xs text-muted mt-3">Using hardcoded USD SOFR-proxy curve. Connect FRED API key for live data.</p>

        <div className="mt-4">
          <Button onClick={() => mutate()} disabled={isPending}>
            <Calculator size={15} />
            {isPending ? 'Pricing…' : 'Price'}
          </Button>
          {error && <p className="text-negative text-sm mt-2">Error: {(error as Error).message}</p>}
        </div>
      </Card>

      {/* Cap / Floor results */}
      {capResult && !isSwaptionTab && !isIrsTab && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="PV ($)" value={`$${capResult.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="PV (bps)" value={capResult.price_bps.toFixed(1)} />
            <Stat label="Strike" value={`${capResult.strike_pct.toFixed(3)}%`} />
            <Stat label="Caplets" value={String(capResult.caplet_details?.length ?? '—')} sub={capResult.pricer_model} />
          </div>
          <Tabs tabs={[{ id: 'strike', label: 'Strike Sensitivity' }, { id: 'vol', label: 'Vol Sensitivity' }, { id: 'cashflows', label: 'Cashflows' }]}>
            {active => (
              <Card>
                {active === 'strike' && (
                  <PlotlyChart data={sensitivityChart(capResult.sensitivity_strike, 'PV vs Strike', '#3b82f6')}
                    layout={{ title: { text: 'Price vs Strike' }, xaxis: { title: { text: 'Strike (%)' } }, yaxis: { title: { text: 'PV ($)' } } }} />
                )}
                {active === 'vol' && (
                  <PlotlyChart data={sensitivityChart(capResult.sensitivity_vol, 'PV vs Vol', '#a78bfa')}
                    layout={{ title: { text: 'Price vs Volatility' }, xaxis: { title: { text: 'Vol' } }, yaxis: { title: { text: 'PV ($)' } } }} />
                )}
                {active === 'cashflows' && (
                  <PlotlyChart
                    data={[{
                      x: capResult.caplet_details.map(d => d.pay_years),
                      y: capResult.caplet_details.map(d => d.pv),
                      type: 'bar', name: 'Caplet PV',
                      marker: { color: '#3b82f6' },
                    }]}
                    layout={{ title: { text: 'Caplet PV by Payment Date' }, xaxis: { title: { text: 'Years' } }, yaxis: { title: { text: 'PV ($)' } } }}
                  />
                )}
              </Card>
            )}
          </Tabs>
        </>
      )}

      {/* Swaption results */}
      {swapResult && isSwaptionTab && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="PV ($)" value={`$${swapResult.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="PV (bps)" value={swapResult.price_bps.toFixed(1)} />
            <Stat label="Par Swap Rate" value={`${swapResult.par_swap_rate_pct.toFixed(3)}%`} />
            <Stat label="Moneyness" value={swapResult.moneyness_label} sub={`${swapResult.moneyness_bps.toFixed(0)} bps · ${swapResult.pricer_model}`} />
          </div>
          <Tabs tabs={[{ id: 'strike', label: 'Strike Sensitivity' }, { id: 'vol', label: 'Vol Sensitivity' }]}>
            {active => (
              <Card>
                {active === 'strike' && (
                  <PlotlyChart data={sensitivityChart(swapResult.sensitivity_strike, 'PV vs Strike', '#3b82f6')}
                    layout={{ title: { text: 'Price vs Strike' }, xaxis: { title: { text: 'Strike (%)' } }, yaxis: { title: { text: 'PV ($)' } } }} />
                )}
                {active === 'vol' && (
                  <PlotlyChart data={sensitivityChart(swapResult.sensitivity_vol, 'PV vs Vol', '#a78bfa')}
                    layout={{ title: { text: 'Price vs Volatility' }, xaxis: { title: { text: 'Vol' } }, yaxis: { title: { text: 'PV ($)' } } }} />
                )}
              </Card>
            )}
          </Tabs>
        </>
      )}

      {/* IRS results */}
      {irsResult && isIrsTab && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="NPV ($)" value={`$${irsResult.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              sub={irsResult.price >= 0 ? 'In your favour' : 'Against you'} />
            <Stat label="NPV (bps)" value={irsResult.price_bps.toFixed(1)} />
            <Stat label="Par Swap Rate" value={`${irsResult.par_swap_rate_pct.toFixed(3)}%`} />
            <Stat label="DV01 ($)" value={`$${irsResult.dv01.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} sub={irsResult.pricer_model} />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Fixed Leg PV" value={`$${irsResult.fixed_leg_pv.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="Float Leg PV" value={`$${irsResult.float_leg_pv.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Stat label="Annuity" value={irsResult.annuity.toFixed(4)} />
          </div>
          <Tabs tabs={[{ id: 'rate', label: 'Rate Sensitivity' }, { id: 'cashflows', label: 'Cashflow Ladder' }]}>
            {active => (
              <Card>
                {active === 'rate' && (
                  <PlotlyChart data={sensitivityChart(irsResult.sensitivity_rate, 'NPV vs Fixed Rate', '#22c55e')}
                    layout={{ title: { text: 'NPV vs Fixed Rate' }, xaxis: { title: { text: 'Fixed Rate (%)' } }, yaxis: { title: { text: 'NPV ($)' } } }} />
                )}
                {active === 'cashflows' && (
                  <PlotlyChart
                    data={[
                      {
                        x: irsResult.leg_details.map(d => d.pay_years),
                        y: irsResult.leg_details.map(d => d.fixed_cashflow),
                        type: 'bar', name: 'Fixed CF',
                        marker: { color: '#3b82f6' },
                      },
                      {
                        x: irsResult.leg_details.map(d => d.pay_years),
                        y: irsResult.leg_details.map(d => d.float_cashflow),
                        type: 'bar', name: 'Float CF',
                        marker: { color: '#22c55e' },
                      },
                    ]}
                    layout={{
                      title: { text: 'Fixed vs Float Cashflows' },
                      barmode: 'group',
                      xaxis: { title: { text: 'Years' } },
                      yaxis: { title: { text: 'Cashflow ($)' } },
                    }}
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
