import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Layers } from 'lucide-react'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import { Select } from '@/components/ui/select'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { fetchVolCubeSwaption } from '@/lib/api'

export default function VolCube() {
  const [ccy, setCcy] = useState('USD')

  const { data, isLoading, error } = useQuery({
    queryKey: ['vol-cube', ccy],
    queryFn: () => fetchVolCubeSwaption(ccy),
    staleTime: 300_000,
  })

  const heatmapData = data ? [{
    z: data.vols_bps as number[][],
    x: data.tenor_grid as string[],
    y: data.expiry_grid as string[],
    type: 'heatmap' as const,
    colorscale: 'Viridis',
    colorbar: { title: 'Vol (bps)' },
  }] : []

  const surfaceData = data ? [{
    z: data.vols_bps as number[][],
    x: data.tenor_grid as string[],
    y: data.expiry_grid as string[],
    type: 'surface' as const,
    colorscale: 'Viridis',
    showscale: true,
    contours: {
      z: { show: true, usecolormap: true, highlightcolor: '#42f462', project: { z: true } },
    },
  }] : []

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Layers size={20} className="text-accent" />
        <div>
          <h1 className="text-xl font-bold text-white">Swaption Vol Cube</h1>
          <p className="text-muted text-sm mt-1">3D normal vol (bps) across expiry × swap tenor</p>
        </div>
      </div>

      <Card>
        <div className="w-48">
          <Select label="Currency" value={ccy} onChange={e => setCcy(e.target.value)}
            options={[{ value: 'USD', label: 'USD' }, { value: 'EUR', label: 'EUR' }, { value: 'GBP', label: 'GBP' }]}
          />
        </div>
      </Card>

      {isLoading && <p className="text-muted text-center py-12">Loading vol cube…</p>}
      {error && <p className="text-negative text-center py-8">Error loading vol cube.</p>}

      {data && (
        <>
          <Card>
            <CardHeader><CardTitle>Heatmap — Expiry × Tenor</CardTitle></CardHeader>
            <PlotlyChart
              data={heatmapData}
              layout={{
                xaxis: { title: { text: 'Swap Tenor' } },
                yaxis: { title: { text: 'Option Expiry' } },
              }}
              style={{ height: 420 }}
            />
          </Card>

          <Card>
            <CardHeader><CardTitle>3D Surface</CardTitle></CardHeader>
            <PlotlyChart
              data={surfaceData}
              layout={{
                scene: {
                  xaxis: { title: 'Swap Tenor' },
                  yaxis: { title: 'Option Expiry' },
                  zaxis: { title: 'Vol (bps)' },
                },
              } as object}
              style={{ height: 500 }}
            />
          </Card>
        </>
      )}
    </div>
  )
}
