import { useState } from 'react'
import { cn } from '@/lib/utils'

interface Tab { id: string; label: string }

interface TabsProps {
  tabs: Tab[]
  children: (activeId: string) => React.ReactNode
  defaultTab?: string
  className?: string
}

export function Tabs({ tabs, children, defaultTab, className }: TabsProps) {
  const [active, setActive] = useState(defaultTab ?? tabs[0]?.id)
  return (
    <div className={className}>
      <div className="flex gap-1 border-b border-border mb-4">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={cn(
              'px-4 py-2 text-sm font-medium transition-colors -mb-px border-b-2',
              active === t.id
                ? 'border-accent text-white'
                : 'border-transparent text-muted hover:text-slate-300'
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
      {children(active)}
    </div>
  )
}
