import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, TrendingUp, BarChart3, Radio, Zap,
  Landmark, Activity, DollarSign, GitCompare, Shield,
  BookOpen, Layers, Gauge, ChevronRight, Repeat2
} from 'lucide-react'
import { cn } from '@/lib/utils'

const nav = [
  { to: '/',                label: 'Home',             icon: LayoutDashboard },
  { to: '/strategies',      label: 'Strategies',        icon: TrendingUp },
  { to: '/vol-surface',     label: 'Vol Surface',       icon: BarChart3 },
  { to: '/live-monitor',    label: 'Live Monitor',      icon: Radio,    badge: 'LIVE' },
  { to: '/vol-strategy',    label: 'Vol Strategy',      icon: Zap },
  { to: '/ir-options',      label: 'IR Options',        icon: Landmark },
  { to: '/delta-hedge',     label: 'Delta Hedge',       icon: Activity },
  { to: '/rates-hub',       label: 'Rates Hub',         icon: DollarSign },
  { to: '/strategy-compare',label: 'Strategy Compare',  icon: GitCompare },
  { to: '/portfolio-risk',  label: 'Portfolio Risk',    icon: Shield },
  { to: '/book-generator',  label: 'Book Generator',    icon: BookOpen },
  { to: '/vol-cube',        label: 'Vol Cube',          icon: Layers },
  { to: '/benchmark',       label: 'Benchmark',         icon: Gauge },
  { to: '/market-making',  label: 'Market Making',     icon: Repeat2, badge: 'NEW' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 flex-shrink-0 bg-sidebar border-r border-border flex flex-col h-screen sticky top-0 overflow-y-auto">
      {/* Logo */}
      <div className="px-4 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded bg-accent flex items-center justify-center">
            <TrendingUp size={14} className="text-white" />
          </div>
          <span className="font-semibold text-sm tracking-wide">Dashboard Pricer</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2">
        {nav.map(({ to, label, icon: Icon, badge }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg mb-0.5 text-sm transition-all group',
                isActive
                  ? 'bg-accent/20 text-white'
                  : 'text-muted hover:text-white hover:bg-panel'
              )
            }
          >
            {({ isActive }) => (
              <>
                <Icon size={15} className={isActive ? 'text-accent' : 'group-hover:text-slate-300'} />
                <span className="flex-1">{label}</span>
                {badge && (
                  <span className="text-[9px] font-bold bg-positive/20 text-positive px-1.5 py-0.5 rounded">
                    {badge}
                  </span>
                )}
                {isActive && <ChevronRight size={12} className="text-accent" />}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="px-4 py-3 border-t border-border text-[11px] text-muted">
        API: <span className="text-slate-400">localhost:8000</span>
      </div>
    </aside>
  )
}
