import { cn } from '@/lib/utils'

type Variant = 'default' | 'positive' | 'negative' | 'warning' | 'blue'

const variants: Record<Variant, string> = {
  default:  'bg-panel text-slate-300 border-border',
  positive: 'bg-positive/20 text-positive border-positive/30',
  negative: 'bg-negative/20 text-negative border-negative/30',
  warning:  'bg-warning/20 text-warning border-warning/30',
  blue:     'bg-accent/20 text-accent border-accent/30',
}

export function Badge({
  children, variant = 'default', className,
}: { children: React.ReactNode; variant?: Variant; className?: string }) {
  return (
    <span className={cn('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border', variants[variant], className)}>
      {children}
    </span>
  )
}
