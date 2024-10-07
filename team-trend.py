import stan
import numpy as np
import polars as pl
from pathlib import Path
from bidict import bidict
TARGET = "PTS"

data_df = []
for p in sorted(Path("data").glob("teams-*.csv")):
    df = pl.read_csv(p)
    year = 2000 + int(p.stem[-2:])
    df = df.with_columns(pl.col("team").str.replace("\*", ""))
    df = df.with_columns(pl.lit(year).alias("year"))
    df = df.select("team", TARGET, "year")
    data_df.append(df)

data_df = pl.concat(data_df)
data_df = data_df.select("team", TARGET, "year")
data_df = data_df.sort("team")
print(data_df)

team_list = list(data_df["team"])
team_dict = [team for team in set(data_df["team"]) if team_list.count(team) == 4]
team_dict = bidict({t: ix for ix, t in enumerate(sorted(team_dict))})
print(team_dict)

K = len(team_dict)
T = max(data_df["year"]) - min(data_df["year"]) + 1
print("k, t", K, T)

X = np.array([list(range(T)) for _ in range(K)]).T
x_tilde = np.ones(K) * T
Y = np.zeros((T, K))

for t, year in enumerate(range(min(data_df["year"]), max(data_df["year"]) +1)):
    print(year)
    df_year = data_df.filter(pl.col("year") == year)
    df_year = df_year.sort("team").select("team", TARGET)

    dict_year = {team: val for (team, val) in zip(df_year["team"], df_year[TARGET])}
    print(dict_year)

    for team, val in dict_year.items():
        if team in team_dict:
            ix_team = team_dict[team]
            print(team, ix_team)
            Y[t, ix_team] = val
        else:
            print("team", team, "not in ", year)
        

with open("models/team-trend.stan") as f:
    teams_code = f.read()


"""
K = 10
T = 4

X = np.array([list(range(T)) for _ in range(K)]).T
print(X)

Y = np.random.randn(T, K)
#Y = Y + X / 1.0
Y = Y / 10.0
print(Y)
"""
teams_data = {"K": K, "T": T, "x": X, "y": Y, "x_tilde": x_tilde}

posterior = stan.build(teams_code, data=teams_data)
fit = posterior.sample(num_chains=4, num_samples=2000)
df = fit.to_frame()  # pandas `DataFrame, requires pandas
print(df)

print(df[["sigma_epsilon", "sigma_alpha"]])

print(df[[col for col in df.columns if "alpha" in col]])
print(df[[col for col in df.columns if "alpha" in col]].mean())
print(df[[col for col in df.columns if "mu" in col]])
print(df[[col for col in df.columns if "mu" in col]].mean())


print(df[[col for col in df.columns if "y_tilde" in col]])
print(df[[col for col in df.columns if "y_tilde" in col]].mean())

tilde_cols = [col for col in df.columns if "y_tilde" in col]
means = {int(tilde.split(".")[-1]) -1: val for tilde, val in zip(tilde_cols, list(df[tilde_cols].mean()))}
means = {team_dict.inv[k]: v for k,v in means.items()}
#print(means)

df = pl.from_pandas(df[tilde_cols])
#print(df)

for col in tilde_cols:
    df = df.with_columns(pl.col(col).alias(team_dict.inv[int(col.split(".")[-1]) -1]))

df = df[list(means)]
print(df)
y_tilde = df.to_numpy()

best = list(np.argmax(y_tilde, axis=1))

best_dict = {}
for b in best:
    best_dict[team_dict.inv[b]] = best_dict.get(team_dict.inv[b],0) + 1.0 / len(best)

best = pl.DataFrame(best_dict)
best = best.transpose(include_header=True, column_names=["p"])
best = best.sort("p")
print(best)
best.write_csv(f"predictions/teams-{TARGET}-winning-probs.csv")

data_2024 = pl.read_csv("data/teams-202324.csv").select("team", "division", "conference")
data_2024 = data_2024.with_columns(pl.col("team").str.replace("\*", ""))
means = df.mean().transpose(include_header=True, column_names=[TARGET], header_name="team")
print(means)

means = means.join(data_2024, on="team", how="left")
means = means.sort("conference", "division", TARGET, descending=True)
print(means)
means.write_csv(f"predictions/teams-{TARGET}.csv")
