import { Link } from 'react-router-dom'
import { TrendingUp, BarChart3, Radio, Zap, Landmark, Activity, DollarSign, GitCompare, Shield, BookOpen, Layers, Gauge } from 'lucide-react'

const pages = [
  { to: '/strategies',       label: 'Strategies',        desc: 'Multi-leg EQD option strategy builder with Greeks', icon: TrendingUp,  color: 'text-blue-400' },
  { to: '/vol-surface',      label: 'Vol Surface',        desc: 'Live SVI / SSVI / Heston surface from Yahoo & Deribit', icon: BarChart3,   color: 'text-purple-400' },
  { to: '/live-monitor',     label: 'Live Monitor',       desc: 'Real-time crypto term structure via WebSocket',        icon: Radio,       color: 'text-green-400', badge: 'LIVE' },
  { to: '/vol-strategy',     label: 'Vol Strategy',       desc: 'ETH/BTC vol premium backtest — short ETH long BTC',   icon: Zap,         color: 'text-yellow-400' },
  { to: '/ir-options',       label: 'IR Options',         desc: 'Caps, floors, swaptions with Bachelier / ZABR',       icon: Landmark,    color: 'text-orange-400' },
  { to: '/delta-hedge',      label: 'Delta Hedge',        desc: 'Delta hedging simulator — gamma vs theta P&L',        icon: Activity,    color: 'text-pink-400' },
  { to: '/rates-hub',        label: 'Rates Hub',          desc: 'Bond pricer, yield curves, FRA, SOFR futures',         icon: DollarSign,  color: 'text-teal-400' },
  { to: '/strategy-compare', label: 'Strategy Compare',   desc: 'Lump sum vs DCA vs All-Weather backtest',             icon: GitCompare,  color: 'text-cyan-400' },
  { to: '/portfolio-risk',   label: 'Portfolio Risk',     desc: 'IR option book Monte Carlo risk & scenario P&L',      icon: Shield,      color: 'text-red-400' },
  { to: '/book-generator',   label: 'Book Generator',     desc: 'Synthetic IR option book — up to 400k positions',     icon: BookOpen,    color: 'text-indigo-400' },
  { to: '/vol-cube',         label: 'Vol Cube',           desc: '3D swaption / cap-floor vol cube heat-map',           icon: Layers,      color: 'text-violet-400' },
  { to: '/benchmark',        label: 'Benchmark',          desc: 'Fast vs QuantLib vs Neural Network pricer comparison', icon: Gauge,       color: 'text-amber-400' },
]

export default function Home() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Dashboard Pricer</h1>
        <p className="text-muted text-sm">
          Equity & interest rate derivatives analytics — React + FastAPI
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {pages.map(({ to, label, desc, icon: Icon, color, badge }) => (
          <Link
            key={to}
            to={to}
            className="bg-surface border border-border rounded-xl p-4 hover:border-accent/50 hover:bg-panel transition-all group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className={`p-2 rounded-lg bg-panel group-hover:bg-surface ${color}`}>
                <Icon size={18} />
              </div>
              {badge && (
                <span className="text-[9px] font-bold bg-positive/20 text-positive px-1.5 py-0.5 rounded">
                  {badge}
                </span>
              )}
            </div>
            <h3 className="text-sm font-semibold text-white mb-1">{label}</h3>
            <p className="text-xs text-muted leading-relaxed">{desc}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}
