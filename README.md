# Option Strategy Visualizer

Educational option pricing environment built with Streamlit.

## Features
- Multi-leg strategies (calls, puts)
- Black-76 pricing
- Greeks visualization
- Data providers:
  - Dummy (educational mode)
  - Yahoo Finance
  - Bloomberg API (optional)

## Run


1. pedagogical purpose, built EQD Options strategyes,
2. Built vol surface from SPX option on yahoo, Calibrate models, heston/dupire sabr on this data, built vol surface for learning.
3. from bbg ability to monitor in live all the option strategies, and build an historical of the Option Strategy entered in a db sqlite
4. Maybe identifyes discrepancies in the vol surface to predict trade ideas

5. Improve vol surface with SVI parametrization

install blpapi
pip install --trusted-host blpapi.bloomberg.com --index-url=http://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi


fix payoff final calendar
essayer de pricer le long avec du monte carlo,...