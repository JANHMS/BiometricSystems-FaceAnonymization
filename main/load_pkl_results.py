import pickle
import pandas as pd

END_PATH = 'representations_deepface.pkl'
DB_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/FERET/probe_10/'
PATH = DB_PATH + END_PATH

unpickled_df = pd.read_pickle(PATH)
print(unpickled_df)
