import { useRef, useState, useCallback } from 'react'
import { Activity } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { LiveChart, type LiveChartRef } from '@/components/charts/LiveChart'
import { useWebSocket } from '@/hooks/useWebSocket'
import { cn, fmt } from '@/lib/utils'

// ── Types ─────────────────────────────────────────────────────────────────────

interface BucketGreeks {
  count: number
  pv: number
  dv01: number
  gamma_up: number
  gamma_dn: number
  vega: number
}

interface StepMsg {
  status: 'init' | 'running' | 'done' | 'error'
  step?: number
  n_steps?: number
  t_label?: string
  shift_bp?: number
  short_rate_pct?: number
  pv?: number
  dv01?: number
  gamma_up?: number
  gamma_dn?: number
  vega?: number
  compute_ms?: number
  n_positions?: number
  pricer_model?: string
  message?: string
  by_expiry?: Record<string, BucketGreeks>
  matrix_dv01?: Record<string, Record<string, number>>
}

type PageStatus = 'idle' | 'connecting' | 'running' | 'paused' | 'done' | 'error'

// ── Constants ─────────────────────────────────────────────────────────────────

const PRICER_MODELS = [
  { value: 'fast',      label: 'Fast (numpy/Bachelier)',   available: true },
  { value: 'quantlib',  label: 'QuantLib (Bachelier)',     available: true },
  { value: 'nn',        label: 'Neural Network',           available: false },
  { value: 'ore',       label: 'ORE (OpenSourceRisk)',     available: false },
  { value: 'strata',    label: 'Strata / Quantra',         available: false },
]

const RATE_MODELS = [
  { value: 'bm', label: 'Brownian Motion (flat σ)' },
  { value: 'ou', label: 'Ornstein-Uhlenbeck (H-W proxy)' },
]

const SPEED_OPTIONS = [
  { value: '50',  label: '50 ms  (fastest)' },
  { value: '100', label: '100 ms' },
  { value: '200', label: '200 ms' },
  { value: '500', label: '500 ms  (QuantLib)' },
]

const STATUS_COLOR: Record<PageStatus, string> = {
  idle:       'text-muted',
  connecting: 'text-yellow-400',
  running:    'text-positive',
  paused:     'text-yellow-400',
  done:       'text-accent',
  error:      'text-negative',
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function IntradayGreeks() {
  // Config
  const [nPositions,    setNPositions]    = useState(300)
  const [seed,          setSeed]          = useState(42)
  const [pricerModel,   setPricerModel]   = useState('fast')
  const [rateModel,     setRateModel]     = useState('ou')
  const [speedMs,       setSpeedMs]       = useState('100')
  const [rateVolBps,    setRateVolBps]    = useState(80)
  const [meanReversion, setMeanReversion] = useState(0.15)
  const [nSteps,        setNSteps]        = useState(390)

  // Runtime state
  const [status,    setStatus]    = useState<PageStatus>('idle')
  const [enabled,   setEnabled]   = useState(false)
  const [wsUrl,     setWsUrl]     = useState('')
  const [current,   setCurrent]   = useState<Partial<StepMsg>>({})
  const [progress,  setProgress]  = useState(0)
  const [byExpiry,  setByExpiry]  = useState<Record<string, BucketGreeks>>({})
  const [matrixDv01, setMatrixDv01] = useState<Record<string, Record<string, number>>>({})

  // Chart refs
  const rateChartRef   = useRef<LiveChartRef>(null)
  const pvChartRef     = useRef<LiveChartRef>(null)
  const dv01ChartRef   = useRef<LiveChartRef>(null)
  const gammaChartRef  = useRef<LiveChartRef>(null)
  const vegaChartRef   = useRef<LiveChartRef>(null)

  const resetCharts = () => {
    rateChartRef.current?.reset()
    pvChartRef.current?.reset()
    dv01ChartRef.current?.reset()
    gammaChartRef.current?.reset()
    vegaChartRef.current?.reset()
  }

  // ── WebSocket message handler ──────────────────────────────────────────────

  const onMessage = useCallback((msg: StepMsg) => {
    if (msg.status === 'init') {
      setStatus('running')
      return
    }
    if (msg.status === 'error') {
      setStatus('error')
      setCurrent(msg)
      return
    }
    if (msg.status === 'done') {
      setStatus('done')
      setEnabled(false)
      setProgress(100)
      return
    }
    if (msg.status === 'running' && msg.t_label !== undefined) {
      const x = msg.t_label!
      rateChartRef.current?.push(x, [msg.shift_bp ?? 0])
      pvChartRef.current?.push(x, [msg.pv ?? 0])
      dv01ChartRef.current?.push(x, [msg.dv01 ?? 0])
      gammaChartRef.current?.push(x, [msg.gamma_up ?? 0, msg.gamma_dn ?? 0])
      vegaChartRef.current?.push(x, [msg.vega ?? 0])
      setCurrent(msg)
      if (msg.by_expiry)   setByExpiry(msg.by_expiry)
      if (msg.matrix_dv01) setMatrixDv01(msg.matrix_dv01)
      if (msg.step !== undefined && msg.n_steps) {
        setProgress(Math.round((msg.step / msg.n_steps) * 100))
      }
    }
  }, [])

  const { status: wsStatus } = useWebSocket<StepMsg>(wsUrl, { onMessage, enabled })

  // ── Controls ───────────────────────────────────────────────────────────────

  const handleStart = () => {
    resetCharts()
    setCurrent({})
    setProgress(0)
    setByExpiry({})
    setMatrixDv01({})
    setStatus('connecting')
    const params = new URLSearchParams({
      n_positions:    String(nPositions),
      seed:           String(seed),
      pricer_model:   pricerModel,
      rate_model:     rateModel,
      n_steps:        String(nSteps),
      speed_ms:       speedMs,
      rate_vol_bps:   String(rateVolBps),
      mean_reversion: String(meanReversion),
    })
    setWsUrl(`/ws/intraday-greeks?${params}`)
    setEnabled(true)
  }

  const handleStop = () => {
    setEnabled(false)
    setStatus('idle')
  }

  const isRunning = status === 'running' || status === 'connecting'

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Activity size={20} className="text-accent" />
        <div>
          <h1 className="text-xl font-bold text-white">Intraday Greeks</h1>
          <p className="text-muted text-sm mt-0.5">
            Simulate one intraday rate path — stream live portfolio Greeks
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className={cn('text-xs font-mono', STATUS_COLOR[status])}>
            {status.toUpperCase()}
          </span>
          {wsStatus === 'open' && (
            <span className="w-2 h-2 rounded-full bg-positive animate-pulse" />
          )}
        </div>
      </div>

      {/* Config + Stats row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Config card */}
        <Card className="lg:col-span-1">
          <CardHeader><CardTitle>Simulation Setup</CardTitle></CardHeader>
          <div className="space-y-3 text-sm">

            <div className="grid grid-cols-2 gap-2">
              <Input label="Positions" type="number" value={nPositions}
                onChange={e => setNPositions(+e.target.value)} min={50} max={5000} step={50} />
              <Input label="Seed" type="number" value={seed}
                onChange={e => setSeed(+e.target.value)} step={1} />
            </div>

            <Select
              label="Pricer model"
              value={pricerModel}
              onChange={e => setPricerModel(e.target.value)}
              options={PRICER_MODELS.map(m => ({
                value: m.value,
                label: m.available ? m.label : `${m.label} — soon`,
              }))}
            />

            <Select
              label="Rate diffusion model"
              value={rateModel}
              onChange={e => setRateModel(e.target.value)}
              options={[
                ...RATE_MODELS,
                { value: 'hw', label: 'Hull-White (calibrated) — soon' },
              ]}
            />

            <Select
              label="Stream speed"
              value={speedMs}
              onChange={e => setSpeedMs(e.target.value)}
              options={SPEED_OPTIONS}
            />

            <div className="grid grid-cols-2 gap-2">
              <Input label="Rate vol (bps ann.)" type="number" value={rateVolBps}
                onChange={e => setRateVolBps(+e.target.value)} min={10} max={300} step={10} />
              <Input label="Mean rev. κ" type="number" value={meanReversion}
                onChange={e => setMeanReversion(+e.target.value)} min={0} max={2} step={0.05} />
            </div>

            <Input label="Steps (mins)" type="number" value={nSteps}
              onChange={e => setNSteps(+e.target.value)} min={10} max={780} step={10} />

            <div className="flex gap-2 pt-1">
              {!isRunning ? (
                <Button onClick={handleStart} className="flex-1">
                  ▶ Start
                </Button>
              ) : (
                <Button onClick={handleStop} className="flex-1" variant="outline">
                  ■ Stop
                </Button>
              )}
              {(status === 'done' || status === 'error') && (
                <Button onClick={() => { resetCharts(); setStatus('idle') }} variant="outline">
                  Reset
                </Button>
              )}
            </div>

            {status === 'error' && (
              <p className="text-negative text-xs mt-1">{current.message}</p>
            )}
          </div>
        </Card>

        {/* Live stats */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>
              Live Greeks
              {current.t_label && (
                <span className="ml-2 text-xs font-mono text-accent">{current.t_label}</span>
              )}
            </CardTitle>
          </CardHeader>

          {/* Progress bar */}
          <div className="h-1 bg-border rounded mb-4">
            <div
              className="h-1 bg-accent rounded transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
            <Stat label="Portfolio PV ($)"
              value={current.pv !== undefined ? fmt(current.pv) : '—'}
              color={current.pv !== undefined ? (current.pv >= 0 ? 'text-positive' : 'text-negative') : undefined}
            />
            <Stat label="DV01 ($)"
              value={current.dv01 !== undefined ? fmt(current.dv01) : '—'}
              color={current.dv01 !== undefined ? (current.dv01 < 0 ? 'text-negative' : 'text-positive') : undefined}
            />
            <Stat label="Vega ($)"
              value={current.vega !== undefined ? fmt(current.vega) : '—'}
            />
            <Stat label="Gamma ↑ ($)"
              value={current.gamma_up !== undefined ? fmt(current.gamma_up) : '—'}
              color="text-positive"
            />
            <Stat label="Gamma ↓ ($)"
              value={current.gamma_dn !== undefined ? fmt(current.gamma_dn) : '—'}
              color="text-negative"
            />
            <Stat label="Rate shift (bps)"
              value={current.shift_bp !== undefined ? current.shift_bp.toFixed(1) : '—'}
            />
          </div>

          <div className="flex gap-4 text-xs text-muted font-mono">
            {current.compute_ms !== undefined && (
              <span>compute: <span className="text-slate-300">{current.compute_ms} ms</span></span>
            )}
            {current.step !== undefined && current.n_steps !== undefined && (
              <span>step: <span className="text-slate-300">{current.step + 1}/{current.n_steps}</span></span>
            )}
            {current.short_rate_pct !== undefined && (
              <span>short rate: <span className="text-slate-300">{current.short_rate_pct.toFixed(3)}%</span></span>
            )}
          </div>
        </Card>
      </div>

      {/* Charts 2×2 + vega */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle>Rate Path (curve shift, bps)</CardTitle></CardHeader>
          <LiveChart
            ref={rateChartRef}
            title=""
            series={[{ name: 'Shift (bps)', color: '#f4a261' }]}
            yTitle="bps"
            zeroLine
            height={190}
          />
        </Card>

        <Card>
          <CardHeader><CardTitle>Portfolio PV ($)</CardTitle></CardHeader>
          <LiveChart
            ref={pvChartRef}
            title=""
            series={[{ name: 'PV', color: '#06d6a0' }]}
            yTitle="$"
            zeroLine
            height={190}
          />
        </Card>

        <Card>
          <CardHeader><CardTitle>DV01 ($)</CardTitle></CardHeader>
          <LiveChart
            ref={dv01ChartRef}
            title=""
            series={[{ name: 'DV01', color: '#00b4d8' }]}
            yTitle="$/bp"
            zeroLine
            height={190}
          />
        </Card>

        <Card>
          <CardHeader><CardTitle>Gamma (±5bp)</CardTitle></CardHeader>
          <LiveChart
            ref={gammaChartRef}
            title=""
            series={[
              { name: 'Γ+ (rates up)',   color: '#22c55e' },
              { name: 'Γ− (rates down)', color: '#ef4444', dash: 'dot' },
            ]}
            yTitle="$"
            zeroLine
            height={190}
          />
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Vega ($  per 1bp normal vol)</CardTitle></CardHeader>
        <LiveChart
          ref={vegaChartRef}
          title=""
          series={[{ name: 'Vega', color: '#a78bfa' }]}
          yTitle="$/bp"
          zeroLine
          height={160}
        />
      </Card>

      {/* ── Greeks ladder by expiry bucket ── */}
      {Object.keys(byExpiry).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Greeks Ladder — by Expiry Bucket</CardTitle>
            {current.t_label && (
              <span className="text-xs font-mono text-accent">{current.t_label}</span>
            )}
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-border text-muted uppercase tracking-wide">
                  {['Expiry', 'Count', 'PV ($)', 'DV01 ($)', 'Γ+ ($)', 'Γ− ($)', 'Vega ($)'].map(h => (
                    <th key={h} className="text-right first:text-left px-3 py-2">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {['<1M','1-3M','3M-1Y','1-2Y','2-5Y','5-10Y','>10Y']
                  .filter(b => byExpiry[b])
                  .map(bucket => {
                    const r = byExpiry[bucket]
                    return (
                      <tr key={bucket} className="border-b border-border/40 hover:bg-panel/50">
                        <td className="px-3 py-1.5 text-slate-300 font-semibold">{bucket}</td>
                        <td className="px-3 py-1.5 text-right text-muted">{r.count}</td>
                        <td className={cn('px-3 py-1.5 text-right', r.pv >= 0 ? 'text-positive' : 'text-negative')}>
                          {r.pv.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className={cn('px-3 py-1.5 text-right', r.dv01 < 0 ? 'text-negative' : 'text-positive')}>
                          {r.dv01.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className="px-3 py-1.5 text-right text-positive">
                          {r.gamma_up.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className="px-3 py-1.5 text-right text-negative">
                          {r.gamma_dn.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className="px-3 py-1.5 text-right text-accent">
                          {r.vega.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                      </tr>
                    )
                  })}
                {/* Total row */}
                {(() => {
                  const rows = Object.values(byExpiry)
                  const tot = {
                    count:    rows.reduce((s, r) => s + r.count, 0),
                    pv:       rows.reduce((s, r) => s + r.pv, 0),
                    dv01:     rows.reduce((s, r) => s + r.dv01, 0),
                    gamma_up: rows.reduce((s, r) => s + r.gamma_up, 0),
                    gamma_dn: rows.reduce((s, r) => s + r.gamma_dn, 0),
                    vega:     rows.reduce((s, r) => s + r.vega, 0),
                  }
                  return (
                    <tr className="border-t-2 border-border font-bold text-white">
                      <td className="px-3 py-2">TOTAL</td>
                      <td className="px-3 py-2 text-right text-muted">{tot.count}</td>
                      <td className={cn('px-3 py-2 text-right', tot.pv >= 0 ? 'text-positive' : 'text-negative')}>
                        {tot.pv.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className={cn('px-3 py-2 text-right', tot.dv01 < 0 ? 'text-negative' : 'text-positive')}>
                        {tot.dv01.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right text-positive">
                        {tot.gamma_up.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right text-negative">
                        {tot.gamma_dn.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className="px-3 py-2 text-right text-accent">
                        {tot.vega.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                    </tr>
                  )
                })()}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── DV01 matrix: expiry × tenor ── */}
      {Object.keys(matrixDv01).length > 0 && (() => {
        const expBuckets = ['<1M','1-3M','3M-1Y','1-2Y','2-5Y','5-10Y','>10Y'].filter(b => matrixDv01[b])
        const tenBuckets = ['≤1Y','1-2Y','2-5Y','5-10Y','>10Y']
        const allTenors  = tenBuckets.filter(t => expBuckets.some(e => matrixDv01[e]?.[t] !== undefined))

        // Heat-map colour: map value to red/green intensity
        const allVals = expBuckets.flatMap(e => allTenors.map(t => matrixDv01[e]?.[t] ?? 0))
        const maxAbs  = Math.max(1, ...allVals.map(Math.abs))
        const cellColor = (v: number) => {
          const intensity = Math.round(Math.min(Math.abs(v) / maxAbs, 1) * 60)
          return v < 0
            ? `rgba(239,68,68,${(intensity / 100).toFixed(2)})`
            : `rgba(34,197,94,${(intensity / 100).toFixed(2)})`
        }

        return (
          <Card>
            <CardHeader>
              <CardTitle>DV01 Matrix — Expiry × Tenor  ($)</CardTitle>
              <span className="text-xs text-muted">Vol-matrix style · red = short, green = long rate risk</span>
            </CardHeader>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-border text-muted uppercase tracking-wide">
                    <th className="text-left px-3 py-2">Expiry ↓ / Tenor →</th>
                    {allTenors.map(t => <th key={t} className="text-right px-3 py-2">{t}</th>)}
                    <th className="text-right px-3 py-2 text-slate-400">Row Σ</th>
                  </tr>
                </thead>
                <tbody>
                  {expBuckets.map(exp => {
                    const rowSum = allTenors.reduce((s, t) => s + (matrixDv01[exp]?.[t] ?? 0), 0)
                    return (
                      <tr key={exp} className="border-b border-border/30 hover:bg-panel/30">
                        <td className="px-3 py-1.5 text-slate-300 font-semibold">{exp}</td>
                        {allTenors.map(t => {
                          const v = matrixDv01[exp]?.[t]
                          return (
                            <td
                              key={t}
                              className="px-3 py-1.5 text-right"
                              style={{ backgroundColor: v !== undefined ? cellColor(v) : undefined }}
                            >
                              {v !== undefined
                                ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
                                : <span className="text-border">—</span>}
                            </td>
                          )
                        })}
                        <td className={cn('px-3 py-1.5 text-right font-bold', rowSum < 0 ? 'text-negative' : 'text-positive')}>
                          {rowSum.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                      </tr>
                    )
                  })}
                  {/* Column totals */}
                  <tr className="border-t-2 border-border font-bold text-white">
                    <td className="px-3 py-2">Col Σ</td>
                    {allTenors.map(t => {
                      const colSum = expBuckets.reduce((s, e) => s + (matrixDv01[e]?.[t] ?? 0), 0)
                      return (
                        <td key={t} className={cn('px-3 py-2 text-right', colSum < 0 ? 'text-negative' : 'text-positive')}>
                          {colSum.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                      )
                    })}
                    <td className={cn('px-3 py-2 text-right', current.dv01 !== undefined && current.dv01 < 0 ? 'text-negative' : 'text-positive')}>
                      {current.dv01?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>
        )
      })()}
    </div>
  )
}
