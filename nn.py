from pathlib import Path
import numpy as np
import pandas as pd
import random
import progressbar
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse

def load_player_stats():
    p = Path("data/")

    dfs = []
    for playerstats in p.glob("20*.csv"):
        df = pd.read_csv(playerstats)
        year = int(str(playerstats.relative_to("data/"))[:4])
        df["year"] = int(str(playerstats.relative_to("data/"))[:4])
        #df = df.head(100)
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.drop("Rk", axis=1)
    return df

def split(df, train=0.7):
    def randomize(elem):
        random.seed(elem)
        if random.random() <= train:
            return "Train"
        else:
            return "Test"

    print(df["Player"])
    df["Split"] = df["Player"].apply(lambda elem: randomize(elem))
    return df

def get_model(x_size=20, h_size=10, hidden_layers=4, out_size=None):
    inputs = keras.Input(shape=(x_size,))
    x = layers.Dense(h_size)(inputs)
    for h in range(hidden_layers):
        x = layers.Dense(h_size, activation="relu")(x)
    if out_size is None:
        outputs = layers.Dense(x_size)(x)
    else:
        outputs = layers.Dense(out_size)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="nhl_network")
    return model

def generate_datasets(df, std, out_column=None, column_legend=None):
    x = []
    y = []
    grouped = df.groupby("Player")
    for group_name, df_group in progressbar.progressbar(grouped):
        df_group = df_group# / std
        df_group = df_group.sort_values("year")
        df_group = df_group.drop("year", axis=1)
        if column_legend is None:
            column_legend = list(df_group.select_dtypes(exclude = ['object']))

        df_group = df_group[column_legend]
        if len(df_group.columns) != len(column_legend):
            break

        df_group = df_group.reset_index(drop=True)
        for ix, row in list(df_group.iterrows())[:-1]:
            row_plus_1 = df_group.loc[ix + 1]

            x.append(np.array(row))
            if out_column is None:
                y.append(np.array(row_plus_1))
            else:
                y.append(row_plus_1[out_column])

    return np.array(x), np.array(y), column_legend

def simulate(df, column, N, std):
    winners = {}

    print("Simulate winners...")
    for n in progressbar.progressbar(range(N)):
        new_df = df[["Player", column]].copy()
        noise = np.random.normal(size=len(df)) * std
        new_df["noise"] = noise
        new_df["simulated"] = new_df[column] + new_df["noise"]

        new_df = new_df.sort_values("simulated")
        winner = list(new_df.tail(1)["Player"])[0]

        winners[winner] = winners.get(winner,0) + 1

    return winners

def main():
    df = load_player_stats()

    values_2021 = df[df["year"] == 2020]
    print("Values for 2021")
    print(values_2021)
    df = split(df)
    std = df.std()

    prediction_variable = "PTS"
    print("prediction_variable", prediction_variable, "mean:", df[prediction_variable].mean())
    print("prediction_variable", prediction_variable, "std:", df[prediction_variable].std())

    mean_absolute_error = (df[prediction_variable] - df[prediction_variable].mean()).abs().mean()
    print("MAE for constant predictor", mean_absolute_error)
    df = df.fillna(0.0)
    x_train, y_train, column_legend = generate_datasets(df, std, out_column=prediction_variable)

    print("XTRAIN", x_train)
    print(column_legend)

    y_naive = x_train[:, column_legend.index(prediction_variable)]
    print("Y naive shape", y_naive.shape)

    naive_error = np.mean(np.abs(y_train - y_naive))
    print("MAE for last-year predictor", naive_error)

    x_train_std = np.std(x_train, axis=0)
    print("np.std(x_train)", x_train_std.shape)
    x_train = x_train / x_train_std
    model = get_model(x_size=len(column_legend), h_size=12, hidden_layers=4, out_size=1)
    keras.utils.plot_model(model, model.name + ".png")
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=["mean_absolute_error"],
    )

    epochs = 200
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_split=0.3, callbacks=[callback])

    x_2021 = values_2021[column_legend]
    x_2021 = x_2021.fillna(0.0)
    print("x 2021")
    print(x_2021)

    print(np.std(x_train))
    x_2021 = x_2021.to_numpy() / x_train_std
    print(x_2021.shape)
    pts_2022 = model.predict(x_2021)
    values_2021[prediction_variable + "2022"] = pts_2022
    print(values_2021)

    values_2021 = values_2021.sort_values(prediction_variable + "2022")
    print(values_2021[["Player", prediction_variable + "2022"]])

    N = 15000
    winners = simulate(values_2021, prediction_variable, N, std[prediction_variable])

    print(winners)
    winners = sorted(winners.items(), key=lambda x: x[1])

    for a, b in winners:
        print(a, b / N)
if __name__ == "__main__":
    main()