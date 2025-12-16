from xbbg import blp

df = blp.bdp("SX5E Index", ["PX_LAST"])
print(df)


ticker = "SX5E 12/19/25 C6000 Index"
df = blp.bdp(ticker, ["PX_LAST", "OPT_DELTA"])
print(df)
