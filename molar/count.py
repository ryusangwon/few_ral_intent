import numpy as np

preds_base = "../tests/predict_base/preds_base.txt"
preds_base = np.loadtxt(preds_base)

preds_contrastive = "../tests/predict_contrastive/preds_contrastive.txt"
preds_contrastive = np.loadtxt(preds_contrastive)

label = "../tests/predict_base/test_labels_np_base"
label = np.loadtxt(label)