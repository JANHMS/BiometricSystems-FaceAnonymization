from deepface import DeepFace
import pandas as pd

# models = [ "DeepFace", "ArcFace" ]
models = [ "DeepFace" ]
# results/blur/5/2.jpg
BASE_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/'
DB_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/FERET/probe_10'
method = 'blur'
k = '20'
i = 2

img_blur = BASE_PATH + 'results' + '/' + method + '/' + k + '/' + str(i) + '.jpg'
# Recognition_Anonymization_Face/FERET/probe/2.jpg
# img_org = BASE_PATH + 'Recognition_Anonymization_Face/FERET/probe/' + str(i) + '.jpg'

for model in models:
    print(f"processing the anonymized picture from {img_blur} and comparing to {DB_PATH}")
    df = DeepFace.find(img_path = img_blur, db_path = DB_PATH, model_name = model)
    df.to_csv('filename.csv')
    