import { useEffect, useRef } from 'react'
import type { Layout, Config } from 'plotly.js'

let Plotly: typeof import('plotly.js-dist-min') | null = null
const loadPlotly = async () => {
  if (!Plotly) Plotly = await import('plotly.js-dist-min')
  return Plotly
}

const DARK_LAYOUT: Partial<Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'rgba(26,29,39,0.5)',
  font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 11 },
  xaxis: { gridcolor: '#2d3354', linecolor: '#2d3354', zerolinecolor: '#3d4471' },
  yaxis: { gridcolor: '#2d3354', linecolor: '#2d3354', zerolinecolor: '#3d4471' },
  legend: { bgcolor: 'rgba(0,0,0,0)', borderwidth: 0 },
  margin: { t: 36, r: 16, b: 48, l: 56 },
}

interface PlotlyChartProps {
  data: object[]          // object[] avoids Plotly strict enum conflicts
  layout?: Partial<Layout>
  config?: Partial<Config>
  style?: React.CSSProperties
  className?: string
}

// Named export alias for backwards-compatibility with existing page imports
export { PlotlyChart }

export default function PlotlyChart({ data, layout, config, style, className }: PlotlyChartProps) {
  const divRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!divRef.current) return
    const el = divRef.current

    loadPlotly().then(plt => {
      plt.react(
        el,
        data as never,
        { ...DARK_LAYOUT, ...layout },
        {
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
          ...config,
        }
      )
    })

    return () => {
      loadPlotly().then(plt => {
        if (el && el.parentNode) plt.purge(el)
      })
    }
  }, [data, layout, config])

  return (
    <div
      ref={divRef}
      className={className}
      style={{ width: '100%', height: 380, ...style }}
    />
  )
}
