from pathlib import Path
import pandas as pd

def load_player_stats():
    p = Path("data/")

    dfs = []
    for playerstats in p.glob("gks20*.csv"):
        print(playerstats)
        df = pd.read_csv(playerstats)
        print(type(playerstats))
        df["year"] = int(str(playerstats.relative_to("data/"))[3:7])
        dfs.append(df)

    df = pd.concat(dfs)
    return df

def main():
    df = load_player_stats()

    keepers = df.groupby("Player").mean()  
    keepers = keepers[keepers["GP"] >= 20]
    print("Best defensive +/-")
    keepers["gk_index"] = keepers["SV%"] + keepers["SV%"]
    keepers = keepers.sort_values("gk_index")
    print(keepers)

if __name__ == "__main__":
    main()