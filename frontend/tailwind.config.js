/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        sidebar: '#0f1117',
        surface: '#1a1d27',
        panel: '#21253a',
        border: '#2d3354',
        muted: '#6b7280',
        accent: '#3b82f6',
        'accent-hover': '#2563eb',
        positive: '#22c55e',
        negative: '#ef4444',
        warning: '#f59e0b',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
