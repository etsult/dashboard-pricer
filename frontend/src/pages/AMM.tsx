import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'
import PlotlyChart from '@/components/charts/PlotlyChart'

// ── Types ─────────────────────────────────────────────────────────────────────

interface AMMMetrics {
  total_pnl: number
  fee_income: number
  il_total: number
  il_vs_hodl: number
  hodl_pnl: number
  rebalance_costs: number
  n_rebalances: number
  time_in_range_pct: number
  daily_pnl_mean: number
  daily_pnl_std: number
  sharpe_daily: number
  max_drawdown_pct: number
  ann_return_pct: number
  initial_capital: number
  final_value: number
  hodl_final_value: number
}

interface BacktestResponse {
  metrics: AMMMetrics
  timestamps: number[]
  mid_prices: number[]
  position_values: number[]
  hodl_values: number[]
  fee_cumulative: number[]
  il_cumulative: number[]
  in_range_flags: boolean[]
}

interface SweepRow {
  range_half_width: number
  range_pct: number
  total_pnl: number
  fee_income: number
  il_total: number
  time_in_range_pct: number
  n_rebalances: number
  sharpe_daily: number
  ann_return_pct: number
}

interface VsMMResponse {
  comparison: Record<string, number | string>
  amm_metrics: AMMMetrics
  mm_metrics: Record<string, number>
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const fmt = (v: number, d = 2) => (v ?? 0).toFixed(d)
const fmtPct = (v: number) => `${(v ?? 0).toFixed(1)}%`
const fmtK = (v: number) => v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${fmt(v)}`

const ASSETS = [
  { symbol: 'SOL/USDT',  sigma: 0.045, vtvl: 1.0 },
  { symbol: 'ETH/USDT',  sigma: 0.030, vtvl: 0.8 },
  { symbol: 'BTC/USDT',  sigma: 0.025, vtvl: 0.5 },
  { symbol: 'AVAX/USDT', sigma: 0.055, vtvl: 1.2 },
  { symbol: 'BNB/USDT',  sigma: 0.035, vtvl: 0.7 },
]

const FEE_TIERS = [
  { label: '0.01%', value: 0.0001 },
  { label: '0.05%', value: 0.0005 },
  { label: '0.30%', value: 0.003  },
  { label: '1.00%', value: 0.01   },
]

// ── Sub-components ────────────────────────────────────────────────────────────

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

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h3 className="text-sm font-semibold text-slate-300 mb-3 mt-6 border-b border-border pb-1">{children}</h3>
}

// ── Main Page ─────────────────────────────────────────────────────────────────

export default function AMM() {
  const [symbol,       setSymbol]       = useState('SOL/USDT')
  const [feeTier,      setFeeTier]      = useState(0.0005)
  const [rangeW,       setRangeW]       = useState(0.10)
  const [capital,      setCapital]      = useState(10000)
  const [vtvl,         setVtvl]         = useState(1.0)
  const [rebalStrat,   setRebalStrat]   = useState('out_of_range')
  const [rebalCost,    setRebalCost]    = useState(10)

  const assetInfo = ASSETS.find(a => a.symbol === symbol) ?? ASSETS[0]

  // ── Mutations ──────────────────────────────────────────────────────────────

  const backtest = useMutation<BacktestResponse, Error>({
    mutationFn: () => axios.post('/api/amm/backtest', {
      symbol, sigma_daily: assetInfo.sigma,
      fee_tier: feeTier, range_half_width: rangeW,
      initial_capital: capital, volume_tvl_ratio: vtvl,
      rebalance_strategy: rebalStrat, rebalance_cost_bps: rebalCost,
      n_bars: 43200,
    }).then(r => r.data),
  })

  const sweep = useMutation<{ rows: SweepRow[] }, Error>({
    mutationFn: () => axios.post('/api/amm/range-sweep', {
      symbol, sigma_daily: assetInfo.sigma,
      fee_tier: feeTier, initial_capital: capital,
      volume_tvl_ratio: vtvl, rebalance_cost_bps: rebalCost,
    }).then(r => r.data),
  })

  const vsMMRes = useMutation<VsMMResponse, Error>({
    mutationFn: () => axios.post('/api/amm/vs-mm', {
      symbol, sigma_daily: assetInfo.sigma,
      fee_tier: feeTier, range_half_width: rangeW,
      initial_capital: capital, volume_tvl_ratio: vtvl,
      rebalance_strategy: rebalStrat, rebalance_cost_bps: rebalCost,
      n_bars: 43200,
    }).then(r => r.data),
  })

  const m = backtest.data?.metrics

  // ── Charts ─────────────────────────────────────────────────────────────────

  const pnlChart = backtest.data ? (() => {
    const d = backtest.data
    const ts = d.timestamps.map(t => new Date(t * 1000).toISOString())
    return [
      { x: ts, y: d.position_values, name: 'LP + Fees', line: { color: '#3b82f6', width: 2 }, type: 'scatter' },
      { x: ts, y: d.hodl_values,     name: '50/50 Hodl', line: { color: '#6b7280', width: 1.5, dash: 'dot' }, type: 'scatter' },
    ]
  })() : []

  const decompositionChart = backtest.data ? (() => {
    const d = backtest.data
    const ts = d.timestamps.map(t => new Date(t * 1000).toISOString())
    return [
      { x: ts, y: d.fee_cumulative, name: 'Cumulative Fees', line: { color: '#22c55e', width: 2 }, type: 'scatter' },
      { x: ts, y: d.il_cumulative,  name: 'IL (realised+open)', line: { color: '#ef4444', width: 2 }, type: 'scatter' },
      { x: ts,
        y: d.fee_cumulative.map((f, i) => f + d.il_cumulative[i]),
        name: 'Net (Fees + IL)', line: { color: '#f59e0b', width: 2 }, type: 'scatter' },
    ]
  })() : []

  const sweepChart = sweep.data ? (() => {
    const rows = sweep.data.rows
    return [
      { x: rows.map(r => r.range_pct), y: rows.map(r => r.total_pnl),
        name: 'Net PnL', mode: 'lines+markers', line: { color: '#3b82f6', width: 2 }, type: 'scatter' },
      { x: rows.map(r => r.range_pct), y: rows.map(r => r.fee_income),
        name: 'Fee Income', mode: 'lines+markers', line: { color: '#22c55e', width: 2, dash: 'dash' }, type: 'scatter' },
      { x: rows.map(r => r.range_pct), y: rows.map(r => r.il_total),
        name: 'IL', mode: 'lines+markers', line: { color: '#ef4444', width: 2, dash: 'dash' }, type: 'scatter' },
    ]
  })() : []

  const inRangeChart = backtest.data ? (() => {
    const d = backtest.data
    const ts = d.timestamps.map(t => new Date(t * 1000).toISOString())
    const inRangeY = d.in_range_flags.map(f => f ? 1 : 0)
    return [
      { x: ts, y: d.mid_prices, name: 'Price', line: { color: '#94a3b8', width: 1.5 }, type: 'scatter', yaxis: 'y' },
      { x: ts, y: inRangeY, name: 'In Range', fill: 'tozeroy', fillcolor: 'rgba(34,197,94,0.15)',
        line: { color: 'transparent' }, type: 'scatter', yaxis: 'y2' },
    ]
  })() : []

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">AMM LP Simulator</h1>
        <p className="text-sm text-muted">
          Uniswap v3 concentrated liquidity · Fee income vs Impermanent Loss · vs AS Order Book MM
        </p>
      </div>

      {/* IL explainer banner */}
      <div className="bg-panel border border-border rounded-lg p-4 mb-6 text-sm text-slate-300">
        <span className="font-semibold text-white">Impermanent Loss (IL)</span>
        {' '}is the underperformance of your LP position vs simply holding the tokens.
        The AMM auto-rebalances (sells rising asset, buys falling asset) — you end up with{' '}
        <span className="text-negative">less of the winner, more of the loser</span>.
        {' '}IL formula: <code className="text-accent font-mono">2√r/(1+r) − 1</code> where{' '}
        <code className="text-accent font-mono">r = P_final/P_entry</code>.{' '}
        Tight ranges amplify IL. Wide ranges reduce IL but dilute fees.
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* ── Controls ── */}
        <div className="xl:col-span-1 space-y-4">
          <div className="bg-panel border border-border rounded-lg p-4">
            <h2 className="text-sm font-semibold text-white mb-4">Parameters</h2>

            {/* Asset */}
            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 block">Asset</span>
              <select value={symbol} onChange={e => setSymbol(e.target.value)}
                className="w-full bg-surface border border-border rounded px-2 py-1.5 text-sm text-white">
                {ASSETS.map(a => <option key={a.symbol} value={a.symbol}>{a.symbol}</option>)}
              </select>
            </label>

            {/* Fee Tier */}
            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 block">Fee Tier</span>
              <select value={feeTier} onChange={e => setFeeTier(Number(e.target.value))}
                className="w-full bg-surface border border-border rounded px-2 py-1.5 text-sm text-white">
                {FEE_TIERS.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
            </label>

            {/* Range width */}
            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 flex justify-between">
                <span>Range half-width (log-price)</span>
                <span className="text-accent">{rangeW.toFixed(2)} (±{((Math.exp(rangeW)-1)*100).toFixed(1)}%)</span>
              </span>
              <input type="range" min={0.02} max={0.80} step={0.01} value={rangeW}
                onChange={e => setRangeW(Number(e.target.value))}
                className="w-full accent-accent" />
            </label>

            {/* Capital */}
            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 block">Initial Capital (USD)</span>
              <input type="number" value={capital} onChange={e => setCapital(Number(e.target.value))}
                className="w-full bg-surface border border-border rounded px-2 py-1.5 text-sm text-white" />
            </label>

            {/* Volume/TVL */}
            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 flex justify-between">
                <span>Volume/TVL ratio (daily)</span>
                <span className="text-accent">{vtvl.toFixed(1)}×</span>
              </span>
              <input type="range" min={0.1} max={5} step={0.1} value={vtvl}
                onChange={e => setVtvl(Number(e.target.value))}
                className="w-full accent-accent" />
            </label>

            {/* Rebalancing */}
            <label className="block mb-3">
              <span className="text-xs text-muted mb-1 block">Rebalancing</span>
              <select value={rebalStrat} onChange={e => setRebalStrat(e.target.value)}
                className="w-full bg-surface border border-border rounded px-2 py-1.5 text-sm text-white">
                <option value="out_of_range">Reset on exit</option>
                <option value="none">No rebalancing</option>
              </select>
            </label>

            {/* Rebalance cost */}
            {rebalStrat === 'out_of_range' && (
              <label className="block mb-3">
                <span className="text-xs text-muted mb-1 flex justify-between">
                  <span>Rebalance cost (bps)</span>
                  <span className="text-accent">{rebalCost} bps</span>
                </span>
                <input type="range" min={1} max={50} step={1} value={rebalCost}
                  onChange={e => setRebalCost(Number(e.target.value))}
                  className="w-full accent-accent" />
              </label>
            )}

            <button onClick={() => backtest.mutate()}
              disabled={backtest.isPending}
              className="w-full bg-accent hover:bg-accent/80 text-white text-sm font-semibold py-2 rounded transition-colors disabled:opacity-50 mt-2">
              {backtest.isPending ? 'Running…' : 'Run Backtest'}
            </button>
            <button onClick={() => sweep.mutate()}
              disabled={sweep.isPending}
              className="w-full bg-surface hover:bg-panel border border-border text-sm text-slate-300 py-2 rounded transition-colors disabled:opacity-50 mt-2">
              {sweep.isPending ? 'Sweeping…' : 'Range Width Sweep'}
            </button>
            <button onClick={() => vsMMRes.mutate()}
              disabled={vsMMRes.isPending}
              className="w-full bg-surface hover:bg-panel border border-border text-sm text-slate-300 py-2 rounded transition-colors disabled:opacity-50 mt-2">
              {vsMMRes.isPending ? 'Comparing…' : 'AMM vs Order Book MM'}
            </button>
          </div>

          {/* Key facts */}
          <div className="bg-panel border border-border rounded-lg p-4 text-xs text-muted space-y-1.5">
            <div><span className="text-slate-400">σ daily:</span> {(assetInfo.sigma * 100).toFixed(1)}%</div>
            <div><span className="text-slate-400">σ per bar:</span> {(assetInfo.sigma / Math.sqrt(1440) * 100).toFixed(4)}%</div>
            <div><span className="text-slate-400">Fee/day (% cap):</span> {(feeTier * vtvl * 100).toFixed(4)}%</div>
            <div><span className="text-slate-400">Optimal w*:</span> {
              (() => {
                const sb = assetInfo.sigma / Math.sqrt(1440)
                const fb = feeTier * vtvl / 1440
                const wopt = fb > 0 ? sb / Math.sqrt(2 * fb) : 0
                return wopt.toFixed(3)
              })()
            } ({(() => {
              const sb = assetInfo.sigma / Math.sqrt(1440)
              const fb = feeTier * vtvl / 1440
              const wopt = fb > 0 ? sb / Math.sqrt(2 * fb) : 0
              return ((Math.exp(wopt)-1)*100).toFixed(0)
            })()}% each side)</div>
          </div>
        </div>

        {/* ── Results ── */}
        <div className="xl:col-span-3 space-y-6">

          {/* Metrics row */}
          {m && (
            <>
              <SectionTitle>P&L Decomposition (30 days)</SectionTitle>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Net PnL" value={fmtK(m.total_pnl)}
                  sub={`${fmtPct(m.ann_return_pct)} annualised`}
                  positive={m.total_pnl > 0} />
                <MetricCard label="Fee Income" value={fmtK(m.fee_income)}
                  sub={`${(m.fee_income / m.initial_capital * 100).toFixed(2)}% of capital`}
                  positive={true} />
                <MetricCard label="IL (total)" value={fmtK(m.il_total)}
                  sub="vs hodl on same entry" positive={false} />
                <MetricCard label="LP vs 50/50 Hodl" value={fmtK(m.il_vs_hodl)}
                  sub={m.il_vs_hodl > 0 ? 'LP beat hodl' : 'Hodl beat LP'}
                  positive={m.il_vs_hodl > 0} />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <MetricCard label="Rebalance Costs" value={fmtK(m.rebalance_costs)}
                  sub={`${m.n_rebalances} rebalances`} />
                <MetricCard label="Time In Range" value={fmtPct(m.time_in_range_pct)} />
                <MetricCard label="Daily PnL μ±σ" value={`$${fmt(m.daily_pnl_mean)} ± $${fmt(m.daily_pnl_std)}`} />
                <MetricCard label="Sharpe (ann.)" value={fmt(m.sharpe_daily, 2)}
                  positive={m.sharpe_daily > 0} />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                <MetricCard label="Initial Capital" value={fmtK(m.initial_capital)} />
                <MetricCard label="Final Value" value={fmtK(m.final_value)}
                  positive={m.final_value > m.initial_capital} />
                <MetricCard label="Max Drawdown" value={fmtPct(m.max_drawdown_pct)} positive={false} />
              </div>
            </>
          )}

          {/* Price + in-range chart */}
          {backtest.data && (
            <>
              <SectionTitle>Price & Range Activity</SectionTitle>
              <PlotlyChart
                data={inRangeChart}
                layout={{
                  yaxis: { title: 'Price', side: 'left' },
                  yaxis2: { title: 'In Range', overlaying: 'y', side: 'right',
                             showgrid: false, range: [0, 3] },
                  height: 220,
                }}
              />
            </>
          )}

          {/* PnL chart */}
          {backtest.data && (
            <>
              <SectionTitle>Portfolio Value: LP vs Hodl</SectionTitle>
              <PlotlyChart data={pnlChart} layout={{ height: 260 }} />
            </>
          )}

          {/* Decomposition chart */}
          {backtest.data && (
            <>
              <SectionTitle>Fee Income vs Impermanent Loss</SectionTitle>
              <PlotlyChart data={decompositionChart} layout={{ height: 260 }} />
            </>
          )}

          {/* Range sweep */}
          {sweep.data && (
            <>
              <SectionTitle>Range Width Sweep — optimal half-width</SectionTitle>
              <PlotlyChart
                data={sweepChart}
                layout={{
                  xaxis: { title: 'Range ±% each side' },
                  yaxis: { title: 'USD PnL (30 days)' },
                  height: 280,
                }}
              />
              <div className="overflow-x-auto mt-3">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-muted border-b border-border">
                      <th className="text-left pb-2 pr-4">Range ±%</th>
                      <th className="text-right pb-2 pr-4">Net PnL</th>
                      <th className="text-right pb-2 pr-4">Fees</th>
                      <th className="text-right pb-2 pr-4">IL</th>
                      <th className="text-right pb-2 pr-4">TIR%</th>
                      <th className="text-right pb-2 pr-4">Rebal.</th>
                      <th className="text-right pb-2">Ann.%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sweep.data.rows.map(row => (
                      <tr key={row.range_half_width}
                        className={`border-b border-border/40 ${row.total_pnl === Math.max(...sweep.data!.rows.map(r => r.total_pnl)) ? 'text-accent' : 'text-slate-300'}`}>
                        <td className="py-1.5 pr-4">±{row.range_pct.toFixed(1)}%</td>
                        <td className={`text-right pr-4 ${row.total_pnl > 0 ? 'text-positive' : 'text-negative'}`}>
                          ${row.total_pnl.toFixed(0)}
                        </td>
                        <td className="text-right pr-4 text-positive">${row.fee_income.toFixed(0)}</td>
                        <td className="text-right pr-4 text-negative">${row.il_total.toFixed(0)}</td>
                        <td className="text-right pr-4">{row.time_in_range_pct.toFixed(0)}%</td>
                        <td className="text-right pr-4">{row.n_rebalances}</td>
                        <td className={`text-right ${row.ann_return_pct > 0 ? 'text-positive' : 'text-negative'}`}>
                          {row.ann_return_pct.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {/* AMM vs MM comparison */}
          {vsMMRes.data && (
            <>
              <SectionTitle>AMM LP vs AS Order Book MM — Same Asset, Same Capital</SectionTitle>
              <div className="grid grid-cols-2 gap-4">
                {/* AMM column */}
                <div className="bg-panel border border-border rounded-lg p-4">
                  <div className="text-xs font-semibold text-muted mb-3 uppercase tracking-wide">AMM LP (Uniswap v3)</div>
                  <div className="space-y-2 text-sm">
                    {[
                      ['Net PnL', `$${fmt(vsMMRes.data.amm_metrics.total_pnl)}`, vsMMRes.data.amm_metrics.total_pnl > 0],
                      ['Fee Income', `$${fmt(vsMMRes.data.amm_metrics.fee_income)}`, true],
                      ['IL Total', `$${fmt(vsMMRes.data.amm_metrics.il_total)}`, false],
                      ['Sharpe', fmt(vsMMRes.data.amm_metrics.sharpe_daily, 2), vsMMRes.data.amm_metrics.sharpe_daily > 0],
                      ['Ann. Return', fmtPct(vsMMRes.data.amm_metrics.ann_return_pct), vsMMRes.data.amm_metrics.ann_return_pct > 0],
                      ['Max Drawdown', fmtPct(vsMMRes.data.amm_metrics.max_drawdown_pct), false],
                    ].map(([label, value, pos]) => (
                      <div key={label as string} className="flex justify-between">
                        <span className="text-muted">{label}</span>
                        <span className={`font-mono ${pos ? 'text-positive' : 'text-negative'}`}>{value}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* MM column */}
                <div className="bg-panel border border-border rounded-lg p-4">
                  <div className="text-xs font-semibold text-muted mb-3 uppercase tracking-wide">Order Book MM (AS Guéant)</div>
                  <div className="space-y-2 text-sm">
                    {[
                      ['Net PnL', `$${fmt(vsMMRes.data.mm_metrics.total_pnl ?? 0)}`, (vsMMRes.data.mm_metrics.total_pnl ?? 0) > 0],
                      ['Spread Income', `$${fmt(vsMMRes.data.mm_metrics.spread_income ?? 0)}`, true],
                      ['Adverse Sel.', `$${fmt(vsMMRes.data.mm_metrics.adverse_selection ?? 0)}`, false],
                      ['Sharpe', fmt(vsMMRes.data.mm_metrics.sharpe_daily ?? 0, 2), (vsMMRes.data.mm_metrics.sharpe_daily ?? 0) > 0],
                      ['Fill Rate', fmtPct(vsMMRes.data.mm_metrics.fill_rate_pct ?? 0), true],
                      ['Max Drawdown', `$${fmt(vsMMRes.data.mm_metrics.max_drawdown ?? 0)}`, false],
                    ].map(([label, value, pos]) => (
                      <div key={label as string} className="flex justify-between">
                        <span className="text-muted">{label}</span>
                        <span className={`font-mono ${pos ? 'text-positive' : 'text-negative'}`}>{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Winner banner */}
              <div className={`mt-3 rounded-lg p-3 text-center text-sm font-semibold border ${
                vsMMRes.data.comparison.winner === 'Order Book MM'
                  ? 'bg-accent/10 border-accent text-accent'
                  : 'bg-positive/10 border-positive text-positive'
              }`}>
                Winner: {vsMMRes.data.comparison.winner as string}
                {' '}· Advantage: ${Math.abs(
                  (vsMMRes.data.comparison.amm_total_pnl as number) -
                  (vsMMRes.data.comparison.mm_total_pnl as number)
                ).toFixed(2)}
              </div>
            </>
          )}

          {/* Theory box */}
          <SectionTitle>Theory: When does AMM LP win?</SectionTitle>
          <div className="bg-panel border border-border rounded-lg p-4 text-xs text-muted space-y-2">
            <p>A Uniswap v3 LP position is equivalent to <span className="text-slate-300">selling a strangle</span> — you earn theta (fees) and lose through gamma (IL). It profits when:</p>
            <p className="font-mono text-slate-300 pl-3">Fee Income &gt; |IL|  →  fee_tier × V/TVL &gt; σ² / (2w)</p>
            <p>Rearranging gives the <span className="text-slate-300">optimal range half-width</span>:</p>
            <p className="font-mono text-accent pl-3">w* = σ_bar / √(2 × fee_bar)   (analogous to AS δ* = σ/√κ)</p>
            <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-border">
              <div><span className="text-slate-400">Low vol + high fees:</span><br/>Tight range profitable</div>
              <div><span className="text-slate-400">High vol + low fees:</span><br/>Wide range or avoid</div>
              <div><span className="text-slate-400">Stablecoin pairs:</span><br/>IL ≈ 0, pure fee capture</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
