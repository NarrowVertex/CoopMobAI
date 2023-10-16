import joblib

study = joblib.load('/content/gdrive/My Drive/Colab_Data/studies/mnist_optuna.pkl')
df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','system_attrs'], axis=1)
df.head(3)
