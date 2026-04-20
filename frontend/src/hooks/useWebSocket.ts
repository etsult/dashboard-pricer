import { useEffect, useRef, useState, useCallback } from 'react'

export type WSStatus = 'connecting' | 'open' | 'closed' | 'error'

interface Options<T> {
  onMessage: (data: T) => void
  enabled?: boolean
}

export function useWebSocket<T>(url: string, { onMessage, enabled = true }: Options<T>) {
  const [status, setStatus] = useState<WSStatus>('closed')
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout>>()
  const retriesRef = useRef(0)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    if (!enabled) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')
    // Build absolute WS URL: replace http(s) with ws(s) and use same host
    const wsUrl = url.startsWith('ws')
      ? url
      : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}${url}`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('open')
      retriesRef.current = 0
    }

    ws.onmessage = (e) => {
      try {
        onMessageRef.current(JSON.parse(e.data) as T)
      } catch {
        // ignore malformed messages
      }
    }

    ws.onerror = () => setStatus('error')

    ws.onclose = () => {
      setStatus('closed')
      if (!enabled) return
      // Exponential back-off: 1s, 2s, 4s, 8s, max 16s
      const delay = Math.min(1000 * 2 ** retriesRef.current, 16_000)
      retriesRef.current += 1
      retryRef.current = setTimeout(connect, delay)
    }
  }, [url, enabled])

  const disconnect = useCallback(() => {
    clearTimeout(retryRef.current)
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  useEffect(() => {
    if (enabled) connect()
    else disconnect()
    return disconnect
  }, [enabled, connect, disconnect])

  return { status, reconnect: connect }
}
