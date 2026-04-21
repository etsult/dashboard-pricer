import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import Home from '@/pages/Home'
import Strategies from '@/pages/Strategies'
import VolSurface from '@/pages/VolSurface'
import LiveMonitor from '@/pages/LiveMonitor'
import VolStrategy from '@/pages/VolStrategy'
import IROptions from '@/pages/IROptions'
import DeltaHedge from '@/pages/DeltaHedge'
import RatesHub from '@/pages/RatesHub'
import StrategyCompare from '@/pages/StrategyCompare'
import PortfolioRisk from '@/pages/PortfolioRisk'
import BookGenerator from '@/pages/BookGenerator'
import VolCube from '@/pages/VolCube'
import Benchmark from '@/pages/Benchmark'
import MarketMaking from '@/pages/MarketMaking'
import AMM from '@/pages/AMM'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="strategies" element={<Strategies />} />
          <Route path="vol-surface" element={<VolSurface />} />
          <Route path="live-monitor" element={<LiveMonitor />} />
          <Route path="vol-strategy" element={<VolStrategy />} />
          <Route path="ir-options" element={<IROptions />} />
          <Route path="delta-hedge" element={<DeltaHedge />} />
          <Route path="rates-hub" element={<RatesHub />} />
          <Route path="strategy-compare" element={<StrategyCompare />} />
          <Route path="portfolio-risk" element={<PortfolioRisk />} />
          <Route path="book-generator" element={<BookGenerator />} />
          <Route path="vol-cube" element={<VolCube />} />
          <Route path="benchmark" element={<Benchmark />} />
          <Route path="market-making" element={<MarketMaking />} />
          <Route path="amm" element={<AMM />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
