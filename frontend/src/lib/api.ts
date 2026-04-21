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

export interface CapFloorRequest {
  curve: { type: 'manual'; points: { tenor: number; rate: number }[] }
  instrument_type: 'cap' | 'floor'
  notional: number; maturity: number; freq: number
  vol_type: 'normal' | 'lognormal'; sigma: number; strike: number
}

export interface CapFloorResponse {
  price: number; price_bps: number; strike_pct: number
  sensitivity_strike: { x: number; price: number }[]
  sensitivity_vol: { x: number; price: number }[]
  caplet_details?: { expiry: number; price: number }[]
}

export interface SwaptionRequest {
  curve: { type: 'manual'; points: { tenor: number; rate: number }[] }
  swaption_type: 'payer' | 'receiver'
  notional: number; expiry: number; swap_tenor: number; freq: number
  vol_type: 'normal' | 'lognormal'; sigma: number; strike: number
}

export interface SwaptionResponse {
  price: number; price_bps: number; par_swap_rate_pct: number
  annuity: number; moneyness_bps: number; moneyness_label: string
  sensitivity_strike: { x: number; price: number }[]
  sensitivity_vol: { x: number; price: number }[]
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

export const generateBook = (req: object) =>
  api.post<BookResponse>('/ir/books/generate', req).then(r => r.data)

export const riskBook = (req: object) =>
  api.post<RiskResponse>('/ir/books/risk', req).then(r => r.data)

export const compareStrategies = (req: CompareRequest) =>
  api.post('/research/backtest/compare', req).then(r => r.data)

export const fetchVolCubeSwaption = (ccy: string) =>
  api.get(`/vol-cube/swaption/${ccy}`).then(r => r.data)
