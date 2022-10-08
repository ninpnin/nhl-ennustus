from pathlib import Path
import numpy as np
import pandas as pd
import random
import progressbar
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import argparse
from probabilistic_loss_functions.losses import *

def get_loss_function(variable, y_mean, y_std, override=None):
    stat_df = pd.read_csv("stat_covariates.csv")
    print(variable, stat_df["stat"])
    print(variable in list(stat_df["stat"]))
    row = stat_df[stat_df["stat"] == variable.strip()].iloc[0]
    print(row)
    model = row["model"]
    print(model)
    if override is not None:
        model = override
    if model == "gaussian":
        return keras.losses.MeanSquaredError(), np.array([y_mean]), lambda x: x
    if model == "gaussian_h":
        return normal_heteroscedastic, np.array([y_mean, tf.math.log(y_std)]), lambda a, b: tf.constant([a])
    if model == "gamma":
        return gamma_loss, np.array([1.0, 1.0]), lambda a, b: tf.math.exp(a - b)
    if model == "poisson":
        return poisson_loss, np.array([tf.math.log(y_mean)]), lambda x: x
    elif model == "neg-binomial":
        # Method of moments init
        r = (y_mean ** 2) / (y_std ** 2 - y_mean)
        p = 1.0 - y_mean / (y_std ** 2)
        return negbin_loss, np.array([r, p]), lambda r, p: tfp.distributions.NegativeBinomial(r, logits=p).mean()
    elif model == "beta":
        return beta_loss, np.array([1.0, 1.0]), lambda a, b: (a + 1) / (a + b + 2)
    
    return None

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
    df = df.drop("-9999", axis=1)
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

def get_model(x_size=20, h_size=10, hidden_layers=4, out_init=None):
    out_size = len(out_init)
    inputs = keras.Input(shape=(x_size,))
    x = tf.keras.layers.Dense(h_size)(inputs)
    for h in range(hidden_layers):
        x = tf.keras.layers.Dense(h_size, activation="relu")(x)
    if out_size is None:
        outputs = tf.keras.layers.Dense(x_size)(x)
    else:
        ki = tf.keras.initializers.RandomNormal(stddev=0.1)
        bi = tf.keras.initializers.Constant(out_init)
        outputs = tf.keras.layers.Dense(out_size, kernel_initializer=ki, bias_initializer=bi)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="nhl_network")
    return model

def build_dataset(df, pred_val, normalize_x=True, normalize_y=False, normalizer=dict(), columns=None):
    mean, std = df.mean(), df.std()
    
    if columns is None:
        columns = sorted(mean.index)
    normalizer["mean"] = normalizer.get("mean", np.array(mean[columns]))
    normalizer["std"] = normalizer.get("std", np.array(std[columns]))
    
    normalizer["mean_y"] = normalizer.get("mean_y", mean[columns.index(pred_val)])
    normalizer["std_y"] = normalizer.get("std_y", std[columns.index(pred_val)])
        
    rows = []
    for player in list(set(df["Player"])):
        df_player = df[df["Player"] == player]
        years = sorted(list(set(df_player["year"])))
        
        for year in years:
            year_p1 = year + 1
            if year_p1 in years:
                row_i = df_player[df_player["year"] == year].iloc[0]
                row_i_p1 = df_player[df_player["year"] == year_p1].iloc[0]
                
                x_i = np.array(row_i[columns])
                y_i_p1 = row_i_p1[pred_val]
                
                if normalize_x:
                    x_i = (x_i - normalizer["mean"]) / normalizer["std"]
                    
                if normalize_y:
                    y_i_p1 = (y_i_p1 - normalizer["mean_y"]) / normalizer["std_y"]
                
                row = list(x_i) + [y_i_p1]
                rows.append(row)
    out_df = pd.DataFrame(rows, columns = columns + ["y"])
    out_df = out_df.fillna(out_df.mean())
    return out_df, columns, normalizer
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--h_size", type=int, default=12)
    parser.add_argument("--h_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pred_var", type=str, default="PP")
    args = parser.parse_args()

    df = load_player_stats()

    future_x = df[df["year"] == args.year - 1]
    future_x = future_x.fillna(future_x.mean())
    df = df[df["year"] != args.year - 1]
    
    # Perform split
    df = split(df)
    train_df, test_df = df[df["Split"] == "Train"], df[df["Split"] == "Test"]
    
    train_df, column_legend, normalizer = build_dataset(train_df, args.pred_var)
    test_df, _, _ = build_dataset(test_df, args.pred_var, columns=column_legend, normalizer=normalizer)
    print(train_df)
    prediction_variable = args.pred_var
    
    y_mean, y_std = df[prediction_variable].mean(), df[prediction_variable].std()
    print("prediction_variable", prediction_variable, "mean:", y_mean)
    print("prediction_variable", prediction_variable, "std:",y_std)

    loss, y_pred_init, mean_func = get_loss_function(args.pred_var, y_mean, y_std, override="gaussian")
    
    #loss, y_pred_size = keras.losses.MeanSquaredError(), np.array([df[prediction_variable].mean()])
    print(loss)
    model = get_model(x_size=len(column_legend), h_size=args.h_size, hidden_layers=args.h_layers, out_init=y_pred_init)
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(lr=0.001),
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)

    x_train = np.array(train_df[column_legend], dtype=np.float32)
    y_train = np.array(train_df["y"], dtype=np.float32)
    print(x_train.shape, y_train.shape)
    history = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_split=0.3, callbacks=[callback])

    x_test = np.array(test_df[column_legend], dtype=np.float32)
    y_test = np.array(test_df["y"], dtype=np.float32)

    eval = model.evaluate(
        x=x_test,
        y=y_test,
    )
    print(eval)
    
    for _, row in test_df.iterrows():
        x_i = (np.array(row[column_legend], dtype=np.float32) - normalizer["mean"]) / normalizer["std"]
        preds_new = model.predict(tf.expand_dims(x_i, axis=0))
        preds_new = preds_new[0]
        if len(y_pred_init) == 2:
            mean_i = mean_func(preds_new[0], preds_new[1])
            mean_i = mean_i.numpy()
        else:
            mean_i = mean_func(preds_new[0])
            
        diff = mean_i - row["y"]
        print(mean_i, row["y"])
        


    means = []
    for _, row in future_x.iterrows():
        x_i = (np.array(row[column_legend], dtype=np.float32) - normalizer["mean"]) / normalizer["std"]
        preds_new = model.predict(tf.expand_dims(x_i, axis=0))
        preds_new = preds_new[0]
        if len(y_pred_init) == 2:
            mean_i = mean_func(preds_new[0], preds_new[1])
            means.append(mean_i.numpy())
        else:
            mean_i = mean_func(preds_new[0])
            means.append(mean_i)
    future_x["y_mean"] = means
    print(future_x)
    future_x.to_csv("new_preds.csv", index=False)

if __name__ == "__main__":
    main()