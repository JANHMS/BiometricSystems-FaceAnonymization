from deepface import DeepFace
import pandas as pd
import json
# 
# all_models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
# test_models = ["DeepFace"]

model = "DeepFace"

NUM_IMG = 5
PROBE_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/FERET/probe'
RESULT_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/results'

# methods = ['blur', 'noise', 'pixelate']
methods = [ 'noise', 'pixelate' ]
# k_factors = [ '5' , '20', '30', '50']
k_factors = ['50']

DB_PATH = '/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/FERET/probe'
# method = 'blur'

for method in methods:
    for k in k_factors:
        scores_mated = list()
        scores_non_mated = list()
        for i in range(2, NUM_IMG):
            try:
                img_anonym = RESULT_PATH + '/' + method + '/' + k + '/' + str(i) + '.jpg'
                for j in range(2, NUM_IMG):                
                    try:
                        img_org = PROBE_PATH + '/' + str(j) + '.jpg'
                        # result = DeepFace.verify("img1.jpg", "img2.jpg", model_name = model)
                        result = DeepFace.verify(img_anonym, img_org, model_name = model)
                        if i == j:
                            scores_mated.append(1 - result["distance"])
                        else:
                            scores_non_mated.append(1 - result["distance"])
                        # df = DeepFace.find(img_path = img_anonym, db_path = PROBE_PATH, model_name = model)
                        print(f"succsessfully worked {i}")
                    except:
                        # print(f"Not found the img_path {img_path}")
                        pass
            except:
                # print(f"can not find path for anonym img {i}")
                pass
        
        with open(f"mated_{method}_{k}.txt", 'w') as f:
            f.write(json.dumps(scores_mated))
        
        with open(f"nonmated_{method}_{k}.txt", 'w') as f:
            f.write(json.dumps(scores_non_mated))
