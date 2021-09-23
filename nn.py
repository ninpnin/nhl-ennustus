from pathlib import Path
import numpy as np
import pandas as pd
import random
import progressbar
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_player_stats():
    p = Path("data/")

    dfs = []
    for playerstats in p.glob("20*.csv"):
        print(playerstats)
        df = pd.read_csv(playerstats)
        print(type(playerstats))
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

def get_model(x_size=20, h_size=10, hidden_layers=4):
    inputs = keras.Input(shape=(x_size,))
    x = layers.Dense(h_size, activation="relu")(inputs)
    for h in range(hidden_layers):
        x = layers.Dense(h_size, activation="relu")(x)
    outputs = layers.Dense(x_size)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="nhl_network")
    return model

def generate_datasets(df, std, column_legend=None):
    x = []
    y = []
    grouped = df.groupby("Player")
    for group_name, df_group in progressbar.progressbar(grouped):
        df_group = df_group / std
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
            y.append(np.array(row_plus_1))

    return np.array(x), np.array(y), column_legend

def main():
    df = load_player_stats()
    df = split(df)
    std = df.std()
    
    """
    print(df)
    print(std)
    train_df = df[df["Split"] == "Train"]
    test_df = df[df["Split"] == "Test"]

    train_df = train_df.fillna(0.0)
    test_df = test_df.fillna(0.0)

    print("Train DF")
    print(train_df)

    print("Test DF")
    print(test_df)

    print(train_df.columns, len(train_df.columns))
    print(test_df.columns, len(test_df.columns))
    
    x_train, y_train, column_legend = generate_datasets(train_df, std)
    x_test, y_test, _ = generate_datasets(test_df, std, column_legend=column_legend)

    #(x_train, y_train), _ = keras.datasets.boston_housing.load_data(
    #    path="boston_housing.npz", test_split=0.2, seed=113
    #)
    print(x_train.shape, y_train.shape)
    """
    df = df.fillna(0.0)
    x_train, y_train, column_legend = generate_datasets(df, std)

    model = get_model(x_size=len(column_legend), h_size=20)
    keras.utils.plot_model(model, model.name + ".png")
    model.compile(
        loss=keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["mean_absolute_error"],
    )

    epochs = 200
    history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_split=0.2)

if __name__ == "__main__":
    main()