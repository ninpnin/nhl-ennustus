import pandas as pd

def main():
    plusminus = pd.read_csv("predictions/+-2022.csv")
    plusminus = plusminus[["Player", "+/-2022"]]
    toi = pd.read_csv("predictions/TOI2022.csv")
    toi = toi[["Player", "TOI2022"]]
    print(plusminus)
    print(toi)

    players = pd.read_csv("data/202122.csv")
    players = players[["Player", "Tm"]]
    print(players)

    players = players.merge(toi, on="Player")
    players = players.merge(plusminus, on="Player")

    print(players)

    byteam = players.groupby("Tm").sum()
    print(byteam)

    players_prime = players.merge(byteam, on="Tm", how="left", suffixes=("", "_sum"))

    print(players_prime)
    players_prime["TOI%"] = players_prime["TOI2022"] / players_prime["TOI2022_sum"]
    print(players_prime)
    players_prime["perf"] = players_prime["TOI%"] * players_prime["+/-2022"]


    teams = players_prime.groupby("Tm")["perf"].sum()
    

    teams = teams.sort_values()

    print(teams)


if __name__ == '__main__':
    
    main()