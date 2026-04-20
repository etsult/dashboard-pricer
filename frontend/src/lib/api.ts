import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Types ──────────────────────────────────────────────────────────────────────

export interface Leg {
  option_type: 'call' | 'put'
  strike: number
  qty: number
  sigma: number
  expiry: string
}

export interface StrategyRequest {
  model: 'Black-76' | 'Black-Scholes' | 'Bachelier'
  forward: number
  rate: number
  dividend_yield: number
  valuation_date: string
  legs: Leg[]
  forward_range_points: number
}

export interface Greeks {
  price: number; delta: number; gamma: number
  vega: number; theta: number; rho: number
}

export interface StrategyResponse {
  greeks: Greeks
  payoff: { forward_range: number[]; payoff_today: number[]; payoff_expiry: number[] }
  greeks_vs_forward: { forward_range: number[]; price: number[]; delta: number[]; gamma: number[]; vega: number[]; theta: number[]; rho: number[] }
}

export interface VolPoint {
  tenor_label: string; days: number; atm_iv_pct: number
  fwd_vol_pct: number | null; fwd_label: string | null
  total_var: number; is_calendar_arb: boolean
}

export interface VolTermStructureResponse {
  currency: string; spot: number; fetched_at: string
  n_quotes: number; n_arb_violations: number
  term_structure: VolPoint[]
}

export interface CurvePoint { tenor: number; tenor_label: string; zero_rate_pct: number; discount_factor: number }
export interface CurveResponse { points: CurvePoint[] }

type CurveManual = { type: 'manual'; points: { tenor: number; rate: number }[] }

export interface ExoticParams {
  pricer_model?: 'fast' | 'quantlib' | 'nn'
  start_shift_y?: number
  day_count?: 'ACT/360' | 'ACT/365' | '30/360'
  settlement_delay_y?: number
  index_key?: string
}

export interface CapFloorRequest extends ExoticParams {
  curve: CurveManual
  instrument_type: 'cap' | 'floor'
  notional: number; maturity: number; freq: number
  vol_type: 'normal' | 'lognormal'; sigma: number; strike: number
}

export interface CapFloorResponse {
  price: number; price_bps: number; strike_pct: number
  caplet_details: { reset_years: number; pay_years: number; fwd_rate_pct: number; discount_factor: number; pv: number }[]
  sensitivity_strike: { x: number; price: number }[]
  sensitivity_vol: { x: number; price: number }[]
  pricer_model: string
}

export interface SwaptionRequest extends ExoticParams {
  curve: CurveManual
  swaption_type: 'payer' | 'receiver'
  notional: number; expiry: number; swap_tenor: number; freq: number
  vol_type: 'normal' | 'lognormal'; sigma: number; strike: number
}

export interface SwaptionResponse {
  price: number; price_bps: number; par_swap_rate_pct: number
  annuity: number; moneyness_bps: number; moneyness_label: string
  sensitivity_strike: { x: number; price: number }[]
  sensitivity_vol: { x: number; price: number }[]
  pricer_model: string
}

export interface IRSRequest {
  curve: CurveManual
  irs_type: 'payer' | 'receiver'
  notional: number
  start_shift_y?: number
  tenor_y: number
  fixed_rate: number
  fixed_freq?: number
  float_freq?: number
  day_count?: 'ACT/360' | 'ACT/365' | '30/360'
  index_key?: string
  pricer_model?: 'fast' | 'quantlib'
  xccy?: boolean
  domestic_ccy?: string
  foreign_ccy?: string
  fx_rate?: number
  basis_spread_bps?: number
}

export interface IRSResponse {
  price: number; price_bps: number; par_swap_rate_pct: number
  annuity: number; fixed_leg_pv: number; float_leg_pv: number; dv01: number
  leg_details: { pay_years: number; fixed_cashflow: number; float_cashflow: number; discount_factor: number; net_pv: number }[]
  sensitivity_rate: { x: number; price: number }[]
  pricer_model: string
}

export interface BookPosition {
  instrument: string; index_key: string; notional: number
  strike: number; expiry_y: number; tenor_y: number
  sigma_n: number; direction: number; label: string
}

export interface BookResponse { book_id: string; n_positions: number; positions: BookPosition[] }

export interface RiskResponse {
  aggregate: {
    total_pv: number; total_dv01: number
    total_gamma_up: number; total_gamma_dn: number
    by_index: Record<string, { pv: number; dv01: number; gamma_up: number; gamma_dn: number }>
    by_expiry: Record<string, { pv: number; dv01: number; gamma_up: number; gamma_dn: number }>
  }
  positions: {
    label: string; instrument: string; index_key: string; ccy: string
    expiry_y: number; tenor_y: number; notional: number; direction: number
    pv: number; dv01: number; gamma_up: number; gamma_dn: number
  }[]
}

export interface IRIndex {
  key: string; label: string; ccy: string
  reset_freq: string; daycount: string; basis_bps: number
}

export interface CompareRequest {
  ticker: string; start: string; end: string
  total_amount: number; dca_frequency: string; transaction_cost_pct: number
}

// ── API helpers ────────────────────────────────────────────────────────────────

export const priceStrategy = (req: StrategyRequest) =>
  api.post<StrategyResponse>('/strategies/price', req).then(r => r.data)

export const fetchVolTermStructure = (currency: string, rate = 0.05) =>
  api.get<VolTermStructureResponse>('/market/vol-term-structure', { params: { currency, rate } }).then(r => r.data)

export const fetchIRIndexes = () =>
  api.get<IRIndex[]>('/ir/indexes').then(r => r.data)

export const priceCapFloor = (req: CapFloorRequest) =>
  api.post<CapFloorResponse>('/ir/cap-floor', req).then(r => r.data)

export const priceSwaption = (req: SwaptionRequest) =>
  api.post<SwaptionResponse>('/ir/swaption', req).then(r => r.data)

export const priceIRS = (req: IRSRequest) =>
  api.post<IRSResponse>('/ir/irs', req).then(r => r.data)

export const generateBook = (req: object) =>
  api.post<BookResponse>('/ir/books/generate', req).then(r => r.data)

export const riskBook = (req: object) =>
  api.post<RiskResponse>('/ir/books/risk', req).then(r => r.data)

export const compareStrategies = (req: CompareRequest) =>
  api.post('/research/backtest/compare', req).then(r => r.data)

export const fetchVolCubeSwaption = (ccy: string) =>
  api.get(`/vol-cube/swaption/${ccy}`).then(r => r.data)

export interface TimingEntryPoint {
  date: string
  final_return_pct: number
  max_drawdown_pct: number
  time_to_recover_days: number | null
  ever_recovered: boolean
}

export interface TimingAnalysisResult {
  ticker: string
  end_date: string
  n_entries: number
  by_entry: TimingEntryPoint[]
  worst_entry: { date: string; entry_price: number; final_return_pct: number; time_to_recover_days: number | null; ever_recovered: boolean }
  best_entry:  { date: string; entry_price: number; final_return_pct: number }
  median_return_pct: number
  pct_entries_positive: number
  worst_dd_path: { dates: string[]; drawdown_pct: number[]; portfolio_value: number[] }
}

export const runTimingAnalysis = (req: {
  ticker: string; start: string; end: string
  total_amount: number; transaction_cost_pct: number; sample_every_n_days?: number
}) => api.post<TimingAnalysisResult>('/research/backtest/timing-analysis', req).then(r => r.data)
