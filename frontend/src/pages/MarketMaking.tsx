import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'
import PlotlyChart from '@/components/charts/PlotlyChart'

// ── Types ─────────────────────────────────────────────────────────────────────

interface MMMetrics {
  total_pnl: number
  spread_income: number
  adverse_selection: number
  n_fills: number
  fill_rate_pct: number
  daily_pnl_mean: number
  daily_pnl_std: number
  sharpe_daily: number
  max_drawdown: number
  avg_quoted_spread_bps: number
}

interface BacktestResponse {
  metrics: MMMetrics
  timestamps: number[]
  mid_prices: number[]
  running_pnl: number[]
  inventories: number[]
  bid_quotes: (number | null)[]
  ask_quotes: (number | null)[]
}

interface SweepRow {
  gamma?: number
  kappa?: number
  target_hs_bps?: number
  total_pnl: number
  daily_pnl_mean: number
  sharpe_daily: number
  n_fills: number
  fill_rate_pct: number
}

const ASSETS = [
  { symbol: 'SOL/USDT',  sigma: 0.045, kappa: 3333, maxInv: 5,   orderSz: 0.1,  price: 150 },
  { symbol: 'ETH/USDT',  sigma: 0.030, kappa: 4000, maxInv: 1,   orderSz: 0.01, price: 3000 },
  { symbol: 'BTC/USDT',  sigma: 0.025, kappa: 5000, maxInv: 0.1, orderSz: 0.001,price: 60000 },
  { symbol: 'AVAX/USDT', sigma: 0.055, kappa: 1666, maxInv: 5,   orderSz: 0.1,  price: 35 },
  { symbol: 'BNB/USDT',  sigma: 0.035, kappa: 2500, maxInv: 5,   orderSz: 0.1,  price: 400 },
]

const fmt = (v: number, d = 2) => (v ?? 0).toFixed(d)
const fmtK = (v: number) => v >= 1000 ? `$${(v/1000).toFixed(1)}k` : `$${fmt(v)}`

function MetricCard({ label, value, sub, positive }: {
  label: string; value: string; sub?: string; positive?: boolean
}) {
  return (
    <div className="bg-panel border border-border rounded-lg p-3">
      <div className="text-[11px] text-muted mb-1">{label}</div>
      <div className={`text-lg font-mono font-semibold ${
        positive === true ? 'text-positive' : positive === false ? 'text-negative' : 'text-white'
      }`}>{value}</div>
      {sub && <div className="text-[10px] text-muted mt-0.5">{sub}</div>}
    </div>
  )
}

export default function MarketMaking() {
  const [symbol,  setSymbol]  = useState('SOL/USDT')
  const [gamma,   setGamma]   = useState(0.10)
  const [kappa,   setKappa]   = useState(3333)
  const [tBars,   setTBars]   = useState(60)

  const asset = ASSETS.find(a => a.symbol === symbol) ?? ASSETS[0]
  const capital = asset.maxInv * asset.price

  const backtest = useMutation<BacktestResponse, Error>({
    mutationFn: () => axios.post('/api/mm/backtest', {
      symbol, sigma_daily: asset.sigma,
      gamma, kappa, T_bars: tBars,
      max_inventory: asset.maxInv, order_size: asset.orderSz,
      n_bars: 43200,
    }).then(r => r.data),
  })

  const gammaSweep = useMutation<{ rows: SweepRow[] }, Error>({
    mutationFn: () => axios.post('/api/mm/sensitivity/gamma', {
      symbol, sigma_daily: asset.sigma, kappa, T_bars: tBars,
      max_inventory: asset.maxInv, order_size: asset.orderSz,
    }).then(r => r.data),
  })

  const kappaSweep = useMutation<{ rows: SweepRow[] }, Error>({
    mutationFn: () => axios.post('/api/mm/sensitivity/kappa', {
      symbol, sigma_daily: asset.sigma, gamma, T_bars: tBars,
      max_inventory: asset.maxInv, order_size: asset.orderSz,
    }).then(r => r.data),
  })

  const m = backtest.data?.metrics

  const pnlChart: object[] = backtest.data ? (() => {
    const d = backtest.data
    const ts = d.timestamps.map(t => new Date(t * 1000).toISOString())
    return [
      { x: ts, y: d.running_pnl, name: 'Running PnL', line: { color: '#3b82f6', width: 2 }, type: 'scatter' },
    ]
  })() : []

  const inventoryChart: object[] = backtest.data ? (() => {
    const d = backtest.data
    const ts = d.timestamps.map(t => new Date(t * 1000).toISOString())
    return [
      { x: ts, y: d.inventories, name: 'Inventory', line: { color: '#f59e0b', width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(245,158,11,0.1)', type: 'scatter' },
    ]
  })() : []

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Market Making — AS Guéant</h1>
        <p className="text-sm text-muted">
          Avellaneda-Stoikov (2008) + Guéant (2013) optimal order book market maker · Fractional space calibration
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Controls */}
        <div className="xl:col-span-1 space-y-4">
          <div className="bg-panel border border-border rounded-lg p-4">
            <h2 className="text-sm font-semibold text-white mb-4">Parameters</h2>

            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 block">Asset</span>
              <select value={symbol} onChange={e => setSymbol(e.target.value)}
                className="w-full bg-surface border border-border rounded px-2 py-1.5 text-sm text-white">
                {ASSETS.map(a => <option key={a.symbol} value={a.symbol}>{a.symbol}</option>)}
              </select>
            </label>

            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 flex justify-between">
                <span>γ (risk aversion)</span>
                <span className="text-accent">{gamma.toFixed(2)}</span>
              </span>
              <input type="range" min={0.01} max={1.0} step={0.01} value={gamma}
                onChange={e => setGamma(Number(e.target.value))} className="w-full accent-accent" />
            </label>

            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 flex justify-between">
                <span>κ (1/target hs fraction)</span>
                <span className="text-accent">{kappa} ({(10000/kappa).toFixed(1)} bps hs)</span>
              </span>
              <input type="range" min={500} max={10000} step={100} value={kappa}
                onChange={e => setKappa(Number(e.target.value))} className="w-full accent-accent" />
            </label>

            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 flex justify-between">
                <span>T_bars (horizon)</span>
                <span className="text-accent">{tBars} bars</span>
              </span>
              <input type="range" min={10} max={240} step={10} value={tBars}
                onChange={e => setTBars(Number(e.target.value))} className="w-full accent-accent" />
            </label>

            <div className="text-xs text-muted mb-3 space-y-1 pt-2 border-t border-border">
              <div>Capital: <span className="text-white">{fmtK(capital)}</span></div>
              <div>σ daily: <span className="text-white">{(asset.sigma*100).toFixed(1)}%</span></div>
              <div>Target hs: <span className="text-white">{(10000/kappa).toFixed(1)} bps</span></div>
            </div>

            <button onClick={() => backtest.mutate()} disabled={backtest.isPending}
              className="w-full bg-accent hover:bg-accent/80 text-white text-sm font-semibold py-2 rounded transition-colors disabled:opacity-50">
              {backtest.isPending ? 'Running…' : 'Run Backtest'}
            </button>
            <button onClick={() => { gammaSweep.mutate(); kappaSweep.mutate() }}
              disabled={gammaSweep.isPending}
              className="w-full bg-surface hover:bg-panel border border-border text-sm text-slate-300 py-2 rounded transition-colors disabled:opacity-50 mt-2">
              {gammaSweep.isPending ? 'Sweeping…' : 'Run Sensitivity Sweeps'}
            </button>
          </div>

          {/* AS formula */}
          <div className="bg-panel border border-border rounded-lg p-4 text-xs text-muted space-y-2">
            <div className="text-slate-300 font-semibold mb-2">AS Optimal Spread Formula</div>
            <div className="font-mono text-accent">δ* = γσ̃²T + (2/γ)ln(1+γ/κ)</div>
            <div className="mt-2">
              <div>Inventory term: <span className="text-white font-mono">{(gamma * (asset.sigma/Math.sqrt(1440))**2 * tBars * 10000).toFixed(2)} bps</span></div>
              <div>Arrival term: <span className="text-white font-mono">{((2/gamma) * Math.log(1 + gamma/kappa) * 10000).toFixed(2)} bps</span></div>
              <div>Full spread ≈ <span className="text-accent font-mono">{((gamma * (asset.sigma/Math.sqrt(1440))**2 * tBars + (2/gamma) * Math.log(1 + gamma/kappa)) * 10000).toFixed(2)} bps</span></div>
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="xl:col-span-3 space-y-6">
          {m && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Total PnL (30d)" value={fmtK(m.total_pnl)} positive={m.total_pnl > 0}
                  sub={`${((m.daily_pnl_mean * 365 / capital) * 100).toFixed(0)}% annualised`} />
                <MetricCard label="Spread Income" value={fmtK(m.spread_income)} positive={true} />
                <MetricCard label="Daily μ ± σ" value={`$${fmt(m.daily_pnl_mean)} ± $${fmt(m.daily_pnl_std)}`} />
                <MetricCard label="Sharpe (ann.)" value={fmt(m.sharpe_daily, 1)} positive={m.sharpe_daily > 0} />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Capital" value={fmtK(capital)} />
                <MetricCard label="Fill Rate" value={`${fmt(m.fill_rate_pct, 1)}%`} />
                <MetricCard label="Fills" value={m.n_fills.toLocaleString()} />
                <MetricCard label="Avg Spread" value={`${fmt(m.avg_quoted_spread_bps)} bps`} />
              </div>
            </>
          )}

          {backtest.data && (
            <>
              <h3 className="text-sm font-semibold text-slate-300 border-b border-border pb-1">Running PnL</h3>
              <PlotlyChart data={pnlChart} layout={{ height: 240 }} />
              <h3 className="text-sm font-semibold text-slate-300 border-b border-border pb-1">Inventory</h3>
              <PlotlyChart data={inventoryChart} layout={{ height: 200 }} />
            </>
          )}

          {/* Gamma sweep */}
          {gammaSweep.data && (
            <>
              <h3 className="text-sm font-semibold text-slate-300 border-b border-border pb-1 mt-4">γ Sensitivity</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-muted border-b border-border">
                      {['γ','PnL','Daily μ','Sharpe','Fills','Fill%'].map(h => (
                        <th key={h} className="text-right pb-2 pr-4 last:pr-0">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {gammaSweep.data.rows.map(row => (
                      <tr key={row.gamma} className="border-b border-border/40 text-slate-300">
                        <td className="py-1.5 pr-4 text-right text-accent">{row.gamma}</td>
                        <td className={`text-right pr-4 ${row.total_pnl > 0 ? 'text-positive' : 'text-negative'}`}>${row.total_pnl.toFixed(0)}</td>
                        <td className="text-right pr-4">${row.daily_pnl_mean.toFixed(2)}</td>
                        <td className="text-right pr-4">{row.sharpe_daily.toFixed(1)}</td>
                        <td className="text-right pr-4">{row.n_fills.toLocaleString()}</td>
                        <td className="text-right">{row.fill_rate_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {/* Kappa sweep */}
          {kappaSweep.data && (
            <>
              <h3 className="text-sm font-semibold text-slate-300 border-b border-border pb-1 mt-4">κ Sensitivity (spread width)</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-muted border-b border-border">
                      {['κ','hs bps','PnL','Sharpe','Fills','Fill%'].map(h => (
                        <th key={h} className="text-right pb-2 pr-4 last:pr-0">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {kappaSweep.data.rows.map(row => {
                      const best = Math.max(...kappaSweep.data!.rows.map(r => r.total_pnl))
                      return (
                        <tr key={row.kappa} className={`border-b border-border/40 ${row.total_pnl === best ? 'text-accent' : 'text-slate-300'}`}>
                          <td className="py-1.5 pr-4 text-right">{row.kappa}</td>
                          <td className="text-right pr-4">{row.target_hs_bps?.toFixed(1)}</td>
                          <td className={`text-right pr-4 ${row.total_pnl > 0 ? 'text-positive' : 'text-negative'}`}>${row.total_pnl.toFixed(0)}</td>
                          <td className="text-right pr-4">{row.sharpe_daily.toFixed(1)}</td>
                          <td className="text-right pr-4">{row.n_fills.toLocaleString()}</td>
                          <td className="text-right">{row.fill_rate_pct.toFixed(1)}%</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {!backtest.data && !gammaSweep.data && (
            <div className="text-center text-muted py-20 text-sm">
              Configure parameters and run the backtest to see results.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
