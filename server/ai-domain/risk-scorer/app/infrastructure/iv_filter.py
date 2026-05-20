def variable_iv(bins):
    d = {}
    for var, df in bins.items():
        try:
            tg = float(df["good"].sum())
            tb = float(df["bad"].sum())
            dg = df["good"] / tg if tg > 0 else 0
            db = df["bad"] / tb if tb > 0 else 0
            iv = float(((db - dg) * df["woe"]).sum())
            d[var] = iv
        except Exception:
            d[var] = 0.0
    return d


def select_by_iv(bins, threshold: float):
    ivs = variable_iv(bins)
    keep = [k for k, v in ivs.items() if v >= threshold]
    return keep
