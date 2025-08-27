import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

def get_feature_names_from_ct(ct: ColumnTransformer):
    output_features = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            last = list(trans.named_steps.values())[-1]
        else:
            last = trans

        if hasattr(last, "get_feature_names_out"):
            feats = last.get_feature_names_out(cols)
        else:
            feats = np.array(cols, dtype=str)
        output_features.extend(feats.tolist())
    return output_features

def shap_values_rf_classifier(rf_model, X):
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        vals = shap_vals[1]
    else:
        vals = shap_vals
    if vals.ndim == 1:
        vals = vals.reshape(1, -1)
    return vals

def shap_barplot_matplotlib(shap_row, feature_names, top_k=10, title="Feature contributions"):
    # Ensure shap_row is 1D
    shap_row = np.array(shap_row).squeeze()

    if shap_row.ndim > 1:
        # e.g., if shape = (features, classes) → take mean across classes
        shap_row = shap_row.mean(axis=-1)

    # Select top absolute contributions
    idx = np.argsort(np.abs(shap_row))[-top_k:][::-1]
    feats = np.array(feature_names, dtype=str)[idx]
    vals = shap_row[idx]

    # Convert feats into a plain Python list of strings
    feats = [str(f) for f in feats]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feats[::-1], vals[::-1])
    ax.set_title(title)
    ax.set_xlabel("SHAP value (impact on output)")
    fig.tight_layout()
    return fig

def explain_prediction(X_row, model, preproc, top_k=10):
    X_proc = preproc.transform(X_row)
    feature_names = get_feature_names_from_ct(preproc)
    shap_vals = shap_values_rf_classifier(model, X_proc)
    shap_row = shap_vals[0]
    fig = shap_barplot_matplotlib(shap_row, feature_names, top_k=top_k)
    return fig