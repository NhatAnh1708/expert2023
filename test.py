from expert_backend import ExpertEyeglassesRecommender
import joblib


# by passing a `lang` parameter you can specify language, which will be used at explanation step
#ins = ExpertEyeglassesRecommender('mesi.jpg', lang='en')
#ins.plot_recommendations()
import pickle

import pickle

# Lưu các đối tượng vào tệp
(bgmm, knn_minority), \
    (bgmj, knn_majority), \
    (bgmj2, knn_majority2), \
    svm, mapper = joblib.load('utils/mixture_plus_svm')
print(svm)