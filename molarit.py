from pathlib import Path
import pandas as pd

def load_player_stats():
    p = Path("data/")

    dfs = []
    for playerstats in list(sorted(p.glob("gks202*.csv"))):
        df = pd.read_csv(playerstats)
        df["year"] = int(str(playerstats.relative_to("data/"))[3:7])
        df["Player"] = [pl.split("\\")[0] for pl in df["Player"]]
        dfs.append(df)

    df = pd.concat(dfs)
    return df

def main():
    df = load_player_stats()

    year = int(df["year"].max()) + 1
    current_keepers = list(df[df["year"] == df["year"].max()]["Player"])

    keepers = df.groupby("Player").mean(numeric_only=True).reset_index()

    keepers = keepers[keepers['Player'].isin(current_keepers)]
    keepers = keepers[keepers["GP"] >= 20]
    #print("Best defensive +/-")
    keepers["gk_index"] = keepers["SV%"] + 0.1 * keepers["GP"] / keepers["GP"].max()
    keepers = keepers.sort_values("gk_index")
    keepers.to_csv(f"predictions/save-percentage-{year}.csv", index=False)

    keepers["GAA_index"] = keepers["GAA"] / keepers["GP"]
    keepers = keepers.sort_values("GAA_index")
    keepers = keepers[["Player", "GP", "GAA_index", "GAA"]]
    keepers.to_csv(f"predictions/GAA_index-{year}.csv", index=False)

if __name__ == "__main__":
    main()