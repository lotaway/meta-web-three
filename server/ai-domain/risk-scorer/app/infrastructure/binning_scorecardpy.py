import scorecardpy as sc

class ScorecardBinning:
    def generate(self, df, target: str):
        return sc.woebin(df, y=target)
    def apply(self, df, bins):
        return sc.woebin_ply(df, bins)

