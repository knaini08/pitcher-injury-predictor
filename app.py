import pandas as pd, numpy as np
from pathlib import Path
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

st.set_page_config(page_title="Pitcher Injury Predictor", layout="wide")

DATA = Path(__file__).resolve().parent / "data"
cands = sorted(DATA.glob("predictions_2023_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not cands:
    st.error("No predictions_2023_*.csv found. Run the notebook G2 cell first.")
    st.stop()

pred_path = cands[0]
preds = pd.read_csv(pred_path).sort_values("injury_prob", ascending=False).reset_index(drop=True)
st.caption(f"Using: {pred_path.name}")

# Optional: enrich names via pybaseball
def add_names(df):
    try:
        from pybaseball import playerid_reverse_lookup
        ids = df["pitcher"].dropna().astype(int).unique().tolist()
        names = playerid_reverse_lookup(ids, key_type="mlbam")
        if "name_first" in names.columns and "name_last" in names.columns:
            names["full_name"] = names["name_first"].str.title() + " " + names["name_last"].str.title()
        elif "name_use" in names.columns:
            names["full_name"] = names["name_use"]
        else:
            name_cols = [c for c in names.columns if "name" in c.lower()]
            names["full_name"] = names[name_cols[0]] if name_cols else ""
        key_col = "key_mlbam" if "key_mlbam" in names.columns else ("mlbam" if "mlbam" in names.columns else None)
        if key_col is None:
            df["player_name"] = df["pitcher"].astype(str)
            return df
        out = df.merge(names[[key_col,"full_name"]], left_on="pitcher", right_on=key_col, how="left")
        out["player_name"] = out["full_name"].fillna(out["pitcher"].astype(str))
        out = out.drop(columns=[c for c in ["key_mlbam","mlbam","full_name"] if c in out.columns])
        return out
    except Exception:
        df["player_name"] = df["pitcher"].astype(str)
        return df

preds = add_names(preds)

y_true = preds["injury_actual"].astype(int).values
y_score = preds["injury_prob"].values
if (y_true.sum() > 0) and (len(y_true) > 0):
    st.sidebar.metric("ROC AUC (2023)", f"{roc_auc_score(y_true, y_score):.3f}")
    try:
        st.sidebar.metric("PR AUC", f"{average_precision_score(y_true, y_score):.3f}")
    except Exception:
        pass

st.title("Pitcher Injury Predictor — 2023")
thr = st.sidebar.slider("Decision threshold", min_value=0.0, max_value=0.2, value=0.01, step=0.001)
k = st.sidebar.number_input("Show Top-K", min_value=5, max_value=200, value=25, step=5)

preds["pred"] = (preds["injury_prob"] >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, preds["pred"]).ravel()
st.sidebar.write("Confusion matrix")
st.sidebar.write(pd.DataFrame({"": ["TN","FP","FN","TP"], "Count":[tn,fp,fn,tp]}))

topk = preds.sort_values("injury_prob", ascending=False).head(int(k)).copy()
topk_display = topk[["player_name","pitcher","injury_prob","injury_actual","pred"]]
topk_display.columns = ["Player","MLBAM","Injury Prob","Actual","Pred"]
st.subheader(f"Top {int(k)} Highest-Risk Pitchers")
st.dataframe(topk_display.style.format({"Injury Prob":"{:.3f}"}), use_container_width=True)