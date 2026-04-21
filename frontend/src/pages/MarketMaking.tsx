import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Activity, TrendingUp, AlertTriangle } from 'lucide-react'
import { Card, CardHeader, CardTitle, Stat } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Tabs } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { api } from '@/lib/api'
import { fmt } from '@/lib/utils'

// ── Types ──────────────────────────────────────────────────────────────────

interface AssetScore {
  symbol: string; exchange: string
  spread_bps: number; daily_volume_m: number
  sigma_daily_pct: number; net_spread_bps: number
  spread_vol_ratio: number; mm_score: number
}

interface BacktestOut {
  symbol: string; model: string; n_bars: number; n_fills: number
  metrics: Record<string, number>
  timestamps: number[]; mid_prices: number[]; inventories: number[]
  running_pnl: number[]; bid_quotes: (number|null)[]; ask_quotes: (number|null)[]
  fills: { timestamp: number; side: string; price: number; qty: number; mid_at_fill: number; spread_quoted: number }[]
}

// ── Main component ─────────────────────────────────────────────────────────

export default function MarketMaking() {
  const [symbol,        setSymbol]       = useState('SOL/USDT')
  const [exchange,      setExchange]     = useState('binance')
  const [model,         setModel]        = useState('AS_Gueant')
  const [gamma,         setGamma]        = useState(0.05)
  const [kappa,         setKappa]        = useState(1.5)
  const [tHours,        setTHours]       = useState(1.0)
  const [maxInv,        setMaxInv]       = useState(5.0)
  const [orderSize,     setOrderSize]    = useState(0.1)
  const [daysSince,     setDaysSince]    = useState(30)
  const [timeframe,     setTimeframe]    = useState('1m')
  const [result,        setResult]       = useState<BacktestOut | null>(null)
  const [sensitivity,   setSensitivity]  = useState<Record<string, number>[]>([])
  const [sensParam,     setSensParam]    = useState<'gamma' | 'kappa'>('gamma')

  // ── Asset scores ────────────────────────────────────────────────────────
  const { data: scores } = useQuery<AssetScore[]>({
    queryKey: ['mm-asset-scores'],
    queryFn: () => api.get('/market-making/asset-scores').then(r => r.data),
    staleTime: Infinity,
  })

  // ── Backtest mutation ────────────────────────────────────────────────────
  const btMut = useMutation({
    mutationFn: () => api.post<BacktestOut>('/market-making/backtest', {
      symbol, exchange, model, since_days_ago: daysSince, timeframe,
      gamma, kappa, T_hours: tHours, max_inventory: maxInv, order_size: orderSize,
      maker_fee: -0.0001, taker_fee: 0.0004, vol_window_bars: 60,
    }).then(r => r.data),
    onSuccess: setResult,
  })

  // ── Sensitivity mutation ─────────────────────────────────────────────────
  const sensMut = useMutation({
    mutationFn: () => api.post<Record<string, number>[]>('/market-making/sensitivity', {
      symbol, exchange, model, since_days_ago: daysSince, timeframe,
      gamma, kappa, T_hours: tHours, max_inventory: maxInv, order_size: orderSize,
      maker_fee: -0.0001, taker_fee: 0.0004, vol_window_bars: 60, param: sensParam,
    }).then(r => r.data),
    onSuccess: setSensitivity,
  })

  const m = result?.metrics ?? {}

  // ── Charts ───────────────────────────────────────────────────────────────
  const ts = result?.timestamps ?? []
  const dates = ts.map(t => new Date(t * 1000).toISOString())

  const pnlChart = result ? [
    { x: dates, y: result.running_pnl, name: 'Running P&L', type: 'scatter', mode: 'lines',
      line: { color: (result.running_pnl[result.running_pnl.length - 1] ?? 0) >= 0 ? '#22c55e' : '#ef4444', width: 2 },
      fill: 'tozeroy' },
  ] : []

  const priceChart = result ? [
    { x: dates, y: result.mid_prices, name: 'Mid Price', type: 'scatter', mode: 'lines',
      line: { color: '#94a3b8', width: 1 } },
    { x: dates, y: result.bid_quotes.map(q => q ?? null), name: 'Bid Quote', type: 'scatter', mode: 'lines',
      line: { color: '#22c55e', width: 1, dash: 'dot' } },
    { x: dates, y: result.ask_quotes.map(q => q ?? null), name: 'Ask Quote', type: 'scatter', mode: 'lines',
      line: { color: '#ef4444', width: 1, dash: 'dot' } },
  ] : []

  const invChart = result ? [
    { x: dates, y: result.inventories, name: 'Inventory', type: 'scatter', mode: 'lines',
      fill: 'tozeroy', line: { color: '#a78bfa', width: 2 } },
  ] : []

  // Sensitivity chart
  const sensKey = sensParam === 'gamma' ? 'gamma' : 'kappa'
  const sensChart = sensitivity.length ? [
    { x: sensitivity.map(r => r[sensKey]), y: sensitivity.map(r => r.total_pnl),
      name: 'Total P&L', type: 'scatter', mode: 'lines+markers', line: { color: '#3b82f6', width: 2 } },
    { x: sensitivity.map(r => r[sensKey]), y: sensitivity.map(r => r.spread_income),
      name: 'Spread Income', type: 'scatter', mode: 'lines+markers', line: { color: '#22c55e', width: 2 } },
    { x: sensitivity.map(r => r[sensKey]), y: sensitivity.map(r => -r.adverse_selection),
      name: '−Adverse Selection', type: 'scatter', mode: 'lines+markers', line: { color: '#ef4444', width: 2, dash: 'dot' } },
  ] : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Activity size={20} className="text-accent" />
        <div>
          <h1 className="text-xl font-bold text-white">Market Making Backtest</h1>
          <p className="text-muted text-sm mt-1">
            Avellaneda-Stoikov (2008) optimal quotes — inventory-adjusted reservation price
          </p>
        </div>
      </div>

      {/* Model explanation */}
      <Card className="border-accent/20 bg-accent/5">
        <div className="text-xs text-slate-300 space-y-1 font-mono">
          <p className="text-accent font-semibold text-sm mb-2">Avellaneda-Stoikov Model</p>
          <p>Reservation price: <span className="text-white">r = s − x·γ·σ²·(T−t)</span></p>
          <p>Optimal spread:    <span className="text-white">δ* = γ·σ²·(T−t) + (2/γ)·ln(1+γ/κ)</span></p>
          <p>Quotes:            <span className="text-white">bid = r − δ*/2 &nbsp;&nbsp; ask = r + δ*/2</span></p>
          <p className="text-muted mt-2">γ = risk aversion · σ = volatility · κ = order arrival rate · x = inventory</p>
        </div>
      </Card>

      {/* Asset scores table */}
      {scores && (
        <Card>
          <CardHeader>
            <CardTitle>Asset Ranking — MM Suitability Score</CardTitle>
            <span className="text-xs text-muted">score = log(vol$) × net_spread × spread_vol_ratio</span>
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead><tr className="border-b border-border">
                {['Symbol', 'Exchange', 'Spread (bps)', 'Volume ($M/day)', 'σ daily', 'Net Spread', 'Spread/Vol', 'Score'].map(h => (
                  <th key={h} className="text-left px-3 py-2 text-muted uppercase">{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {scores.map((a, i) => (
                  <tr
                    key={a.symbol}
                    className={`border-b border-border/40 hover:bg-panel/50 cursor-pointer ${symbol === a.symbol ? 'bg-accent/10' : ''}`}
                    onClick={() => { setSymbol(a.symbol); setExchange(a.exchange.toLowerCase()) }}
                  >
                    <td className="px-3 py-2 font-mono font-semibold text-white">{a.symbol}</td>
                    <td className="px-3 py-2 text-muted">{a.exchange}</td>
                    <td className="px-3 py-2 font-mono text-blue-400">{a.spread_bps.toFixed(1)}</td>
                    <td className="px-3 py-2 font-mono">{a.daily_volume_m.toLocaleString()}</td>
                    <td className="px-3 py-2 font-mono">{a.sigma_daily_pct.toFixed(1)}%</td>
                    <td className="px-3 py-2 font-mono text-green-400">{a.net_spread_bps.toFixed(1)}</td>
                    <td className="px-3 py-2 font-mono text-purple-400">{a.spread_vol_ratio.toFixed(3)}</td>
                    <td className="px-3 py-2 font-mono font-bold">
                      <Badge variant={i === 0 ? 'positive' : i < 3 ? 'blue' : 'default'}>
                        {a.mm_score.toFixed(1)}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted mt-2">Click a row to select that asset.</p>
        </Card>
      )}

      {/* Strategy parameters */}
      <Card>
        <CardHeader><CardTitle>Strategy Parameters</CardTitle></CardHeader>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Input label="Symbol" value={symbol} onChange={e => setSymbol(e.target.value)} placeholder="SOL/USDT" />
          <Select label="Exchange" value={exchange} onChange={e => setExchange(e.target.value)}
            options={[{ value: 'binance', label: 'Binance' }, { value: 'bybit', label: 'Bybit' }, { value: 'okx', label: 'OKX' }]} />
          <Select label="Model" value={model} onChange={e => setModel(e.target.value)}
            options={[{ value: 'AS_basic', label: 'AS Basic (2008)' }, { value: 'AS_Gueant', label: 'AS + Guéant (2013)' }]} />
          <Select label="Timeframe" value={timeframe} onChange={e => setTimeframe(e.target.value)}
            options={[{ value: '1m', label: '1 min' }, { value: '5m', label: '5 min' }, { value: '15m', label: '15 min' }]} />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          <Input label="γ Risk Aversion" type="number" value={gamma} onChange={e => setGamma(+e.target.value)} step="0.01" min="0.001" />
          <Input label="κ Order Arrival" type="number" value={kappa} onChange={e => setKappa(+e.target.value)} step="0.1" min="0.1" />
          <Input label="T Horizon (hrs)" type="number" value={tHours} onChange={e => setTHours(+e.target.value)} step="0.5" min="0.1" />
          <Input label="Order Size" type="number" value={orderSize} onChange={e => setOrderSize(+e.target.value)} step="0.01" min="0.001" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          <Input label="Max Inventory" type="number" value={maxInv} onChange={e => setMaxInv(+e.target.value)} step="1" />
          <Input label="Backtest Days" type="number" value={daysSince} onChange={e => setDaysSince(+e.target.value)} step="7" min="1" max="365" />
        </div>
        <div className="flex gap-3 mt-4 flex-wrap">
          <Button onClick={() => btMut.mutate()} disabled={btMut.isPending}>
            <TrendingUp size={15} />
            {btMut.isPending ? 'Running backtest…' : 'Run Backtest'}
          </Button>
          <div className="flex items-center gap-2">
            <Select label="" value={sensParam} onChange={e => setSensParam(e.target.value as 'gamma' | 'kappa')}
              options={[{ value: 'gamma', label: 'γ sweep' }, { value: 'kappa', label: 'κ sweep' }]} />
            <Button variant="outline" onClick={() => sensMut.mutate()} disabled={sensMut.isPending} className="mt-5">
              {sensMut.isPending ? 'Sweeping…' : 'Parameter Sweep'}
            </Button>
          </div>
        </div>
        {btMut.error && (
          <div className="mt-3 flex items-center gap-2 text-warning text-sm">
            <AlertTriangle size={14} />
            Exchange unreachable — using synthetic data for demo.
          </div>
        )}
      </Card>

      {/* Results */}
      {result && (
        <>
          {/* P&L decomposition */}
          <div>
            <p className="text-xs text-muted uppercase tracking-wide mb-2">P&L Decomposition</p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Total P&L" value={`$${fmt(m.total_pnl, 2)}`}
                color={m.total_pnl >= 0 ? 'text-positive' : 'text-negative'} />
              <Stat label="Spread Income" value={`$${fmt(m.spread_income, 2)}`} color="text-positive" />
              <Stat label="Adverse Sel." value={`$${fmt(m.adverse_selection, 2)}`} color="text-negative"
                sub={`${(m.adverse_sel_ratio * 100).toFixed(0)}% of gross`} />
              <Stat label="Inventory PnL" value={`$${fmt(m.inventory_pnl, 2)}`}
                color={m.inventory_pnl >= 0 ? 'text-positive' : 'text-negative'} />
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Sharpe (ann.)" value={fmt(m.sharpe_annualised)} />
            <Stat label="Max Drawdown" value={`$${fmt(m.max_drawdown, 2)}`} color="text-negative" />
            <Stat label="Fills" value={m.n_fills}
              sub={`${m.n_bid_fills} bid / ${m.n_ask_fills} ask`} />
            <Stat label="P&L per $1M vol" value={`$${fmt(m.pnl_per_1m_volume, 0)}`} />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Spread Capture" value={`${(m.spread_capture_ratio * 100).toFixed(0)}%`}
              color={m.spread_capture_ratio > 0.5 ? 'text-positive' : 'text-warning'} />
            <Stat label="Max Inventory" value={fmt(m.max_inventory, 3)} />
            <Stat label="TWAI" value={fmt(m.twai, 3)} sub="time-wtd avg |inventory|" />
            <Stat label="Vol Traded ($)" value={`$${(m.total_volume_quote ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
          </div>

          <Tabs tabs={[
            { id: 'pnl',    label: 'P&L' },
            { id: 'quotes', label: 'Quotes & Price' },
            { id: 'inv',    label: 'Inventory' },
          ]}>
            {active => (
              <Card>
                {active === 'pnl' && (
                  <PlotlyChart data={pnlChart}
                    layout={{ yaxis: { title: { text: 'Running P&L (USDT)' } }, xaxis: { title: { text: 'Date' } } }} />
                )}
                {active === 'quotes' && (
                  <PlotlyChart data={priceChart}
                    layout={{ yaxis: { title: { text: 'Price' } }, xaxis: { title: { text: 'Date' } } }} />
                )}
                {active === 'inv' && (
                  <PlotlyChart data={invChart}
                    layout={{ yaxis: { title: { text: 'Inventory (base asset)' } }, xaxis: { title: { text: 'Date' } } }} />
                )}
              </Card>
            )}
          </Tabs>
        </>
      )}

      {/* Sensitivity analysis */}
      {sensitivity.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Parameter Sensitivity — {sensParam === 'gamma' ? 'γ' : 'κ'} sweep</CardTitle>
          </CardHeader>
          <PlotlyChart
            data={sensChart}
            layout={{
              xaxis: { title: { text: sensParam === 'gamma' ? 'γ (risk aversion)' : 'κ (order arrival)' } },
              yaxis: { title: { text: 'P&L (USDT)' } },
            }}
          />
          <div className="overflow-x-auto mt-4">
            <table className="w-full text-xs">
              <thead><tr className="border-b border-border">
                {[sensParam, 'Total PnL', 'Spread Income', 'Adverse Sel.', 'Sharpe', 'Max DD', 'Fills'].map(h => (
                  <th key={h} className="text-left px-3 py-2 text-muted uppercase">{h}</th>
                ))}
              </tr></thead>
              <tbody>
                {sensitivity.map((row, i) => (
                  <tr key={i} className="border-b border-border/40 hover:bg-panel/50">
                    <td className="px-3 py-2 font-mono font-bold text-accent">{fmt(row[sensParam], 3)}</td>
                    <td className={`px-3 py-2 font-mono ${row.total_pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{fmt(row.total_pnl, 2)}</td>
                    <td className="px-3 py-2 font-mono text-green-400">{fmt(row.spread_income, 2)}</td>
                    <td className="px-3 py-2 font-mono text-red-400">{fmt(row.adverse_selection, 2)}</td>
                    <td className="px-3 py-2 font-mono">{fmt(row.sharpe_annualised, 2)}</td>
                    <td className="px-3 py-2 font-mono text-negative">{fmt(row.max_drawdown, 2)}</td>
                    <td className="px-3 py-2 font-mono">{row.n_fills}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
