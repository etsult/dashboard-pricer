import { cn } from '@/lib/utils'

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  options: { value: string; label: string }[]
}

export function Select({ label, options, className, id, ...props }: SelectProps) {
  const selectId = id ?? label?.toLowerCase().replace(/\s+/g, '-')
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label htmlFor={selectId} className="text-xs text-muted uppercase tracking-wide">
          {label}
        </label>
      )}
      <select
        id={selectId}
        className={cn(
          'h-9 w-full rounded-md border border-border bg-panel px-3 text-sm text-slate-200',
          'focus:outline-none focus:ring-2 focus:ring-accent',
          'disabled:opacity-40 cursor-pointer',
          className
        )}
        {...props}
      >
        {options.map(o => (
          <option key={o.value} value={o.value} className="bg-surface">
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}
