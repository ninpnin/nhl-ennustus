from pathlib import Path
import pandas as pd

def load_player_stats():
    p = Path("data/")

    dfs = []
    for playerstats in p.glob("20*.csv"):
        print(playerstats)
        df = pd.read_csv(playerstats)
        print(type(playerstats))
        df["year"] = int(str(playerstats.relative_to("data/"))[:4])
        dfs.append(df)

    df = pd.concat(dfs)
    return df

def get_model(x_size=20, h_size=10):
    inputs = keras.Input(shape=(784,))
    h1 = layers.Dense(h_size, activation="relu")(inputs)
    h2 = layers.Dense(h_size, activation="relu")(h1)
    outputs = layers.Dense(x_size)(h2)
    model = keras.Model(inputs=inputs, outputs=outputs, name="nhl_network")
    return model

def main():
    df = load_player_stats()
    std = df.std()
    print(df)
    print(std)

    normalized = df / std
    print(normalized)

    
if __name__ == "__main__":
    main()