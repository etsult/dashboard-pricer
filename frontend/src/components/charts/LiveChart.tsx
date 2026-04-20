import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react'

let _Plotly: typeof import('plotly.js-dist-min') | null = null
async function getPlotly() {
  if (!_Plotly) _Plotly = await import('plotly.js-dist-min')
  return _Plotly
}

const DARK: Record<string, unknown> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'rgba(26,29,39,0.5)',
  font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 10 },
  xaxis: { gridcolor: '#2d3354', linecolor: '#2d3354', zerolinecolor: '#3d4471' },
  yaxis: { gridcolor: '#2d3354', linecolor: '#2d3354', zerolinecolor: '#3d4471' },
  legend: { bgcolor: 'rgba(0,0,0,0)', borderwidth: 0, font: { size: 10 } },
  margin: { t: 28, r: 10, b: 36, l: 60 },
}

export interface LiveChartRef {
  push(x: string, ys: number[]): void
  reset(): void
}

interface Series { name: string; color: string; dash?: string }

interface Props {
  title: string
  series: Series[]
  yTitle?: string
  height?: number
  maxPoints?: number
  zeroLine?: boolean
  className?: string
}

export const LiveChart = forwardRef<LiveChartRef, Props>(function LiveChart(
  { title, series, yTitle, height = 190, maxPoints = 500, zeroLine = false, className },
  ref,
) {
  const divRef  = useRef<HTMLDivElement>(null)
  const pltRef  = useRef<typeof import('plotly.js-dist-min') | null>(null)
  const initRef = useRef(false)

  const makeTraces = () =>
    series.map(s => ({
      x: [] as string[],
      y: [] as number[],
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: s.name,
      line: { color: s.color, width: 1.5, dash: s.dash ?? 'solid' },
    }))

  const makeLayout = () => ({
    ...DARK,
    title: { text: title, font: { size: 11, color: '#cbd5e1' } },
    yaxis: {
      ...(DARK.yaxis as object),
      title: { text: yTitle ?? '', font: { size: 10 } },
      zerolinecolor: zeroLine ? '#6b7280' : '#3d4471',
    },
    showlegend: series.length > 1,
  })

  useEffect(() => {
    if (initRef.current || !divRef.current) return
    initRef.current = true
    getPlotly().then(plt => {
      pltRef.current = plt
      plt.newPlot(divRef.current!, makeTraces(), makeLayout(), {
        responsive: true,
        displayModeBar: false,
      })
    })
    return () => {
      getPlotly().then(plt => {
        if (divRef.current) plt.purge(divRef.current)
      })
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useImperativeHandle(ref, () => ({
    push(x: string, ys: number[]) {
      const plt = pltRef.current
      if (!plt || !divRef.current) return
      plt.extendTraces(
        divRef.current,
        { x: ys.map(() => [x]), y: ys.map(y => [y]) },
        ys.map((_, i) => i),
        maxPoints,
      )
    },
    reset() {
      const plt = pltRef.current
      if (!plt || !divRef.current) return
      plt.react(divRef.current!, makeTraces(), makeLayout())
    },
  }))

  return (
    <div ref={divRef} className={className} style={{ width: '100%', height }} />
  )
})
