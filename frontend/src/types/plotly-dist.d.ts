// plotly.js-dist-min doesn't ship its own types — re-use the full plotly.js types
declare module 'plotly.js-dist-min' {
  export * from 'plotly.js'
  export { default } from 'plotly.js'
}
