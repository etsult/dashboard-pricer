import { forwardRef } from 'react'
import { cn } from '@/lib/utils'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, id, ...props }, ref) => {
    const inputId = id ?? label?.toLowerCase().replace(/\s+/g, '-')
    return (
      <div className="flex flex-col gap-1">
        {label && (
          <label htmlFor={inputId} className="text-xs text-muted uppercase tracking-wide">
            {label}
          </label>
        )}
        <input
          id={inputId}
          ref={ref}
          className={cn(
            'h-9 w-full rounded-md border border-border bg-panel px-3 text-sm text-slate-200',
            'placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent',
            'disabled:opacity-40 disabled:cursor-not-allowed',
            className
          )}
          {...props}
        />
      </div>
    )
  }
)
Input.displayName = 'Input'
