import pickle
import numpy as np
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

with open("../GraPyTor/WEDA-FALL/models/xgboost_pond_trial_has_fall.pkl", "rb") as f:
    booster = pickle.load(f)

print("Type du modèle chargé :", type(booster))

n_features = booster.num_features()
print("Nombre de features :", n_features)

initial_type = [('float_input', FloatTensorType([None, n_features]))]

onnx_model = convert_xgboost(booster, initial_types=initial_type)

with open("./conversion/xgboost_pond_trial_has_fall.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Conversion XGBoost -> ONNX terminée")