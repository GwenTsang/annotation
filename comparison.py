#  Comparaison inter-runs (exemple : avec vs sans annotations d'experts)

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Chemins ─────────────────────────────────────────────────────────────────
JSONL_RUN1 = "/content/a/outputs/homophobie/homophobie_scenario_julie_run001.jsonl"
JSONL_RUN2 = "/content/a/outputs/homophobie/homophobie_scenario_julie_run002.jsonl"
ORIG_XLSX  = "/content/a/data/homophobie_scenario_julie.xlsx"  # pour récupérer TEXT
OUT_DIR    = "/content/a/outputs/homophobie/comparaison_runs"
os.makedirs(OUT_DIR, exist_ok=True)

EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie",
]

# ════════════════════════════════════════════════════════════════════════════
#  1. CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════

def load_emotions_from_jsonl(path: str) -> pd.DataFrame:
    """Charge un JSONL et retourne un DataFrame idx × 11 émotions."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = {"idx": rec["idx"], "row_id": rec.get("row_id"), "json_ok": rec.get("json_ok", False)}
            pj = rec.get("parsed_json")
            if rec.get("json_ok") and isinstance(pj, dict):
                emo = pj.get("emotions", {})
                for e in EMOTIONS:
                    row[e] = emo.get(e, np.nan)
                row["confidence"] = pj.get("metadata", {}).get("confidence")
                row["rationale"]  = pj.get("rationale_short")
            else:
                for e in EMOTIONS:
                    row[e] = np.nan
                row["confidence"] = None
                row["rationale"]  = None
            rows.append(row)
    return pd.DataFrame(rows)

df1 = load_emotions_from_jsonl(JSONL_RUN1)
df2 = load_emotions_from_jsonl(JSONL_RUN2)
print(f"Run 1 (avec annotations)  : {len(df1)} lignes")
print(f"Run 2 (sans annotations)  : {len(df2)} lignes")

# ── Texte original (pour afficher les divergences) ─────────────────────────
df_text = None
if ORIG_XLSX and os.path.exists(ORIG_XLSX):
    df_text = pd.read_excel(ORIG_XLSX).reset_index(drop=True)
    print(f"XLSX original chargé      : {len(df_text)} lignes")

# ════════════════════════════════════════════════════════════════════════════
#  2. ALIGNEMENT DES DEUX RUNS
# ════════════════════════════════════════════════════════════════════════════

# Suffixes pour distinguer les colonnes
merged = pd.merge(
    df1, df2,
    on="idx", how="inner",
    suffixes=("_r1", "_r2"),
)
# Garder uniquement les lignes où les DEUX runs ont un JSON valide
merged = merged[merged["json_ok_r1"] & merged["json_ok_r2"]].copy()
N = len(merged)
print(f"\n✓ {N} messages comparables (JSON valide dans les deux runs)")

# ════════════════════════════════════════════════════════════════════════════
#  3. MÉTRIQUES D'ACCORD PAR ÉMOTION
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  ACCORD PAR ÉMOTION (run1 = avec experts  vs  run2 = sans experts)")
print(f"{'='*70}\n")

stats_rows = []
for e in EMOTIONS:
    c1 = f"{e}_r1"
    c2 = f"{e}_r2"
    v1 = merged[c1].values.astype(int)
    v2 = merged[c2].values.astype(int)

    agree     = (v1 == v2).sum()
    agree_pct = agree / N * 100
    both_1    = ((v1 == 1) & (v2 == 1)).sum()
    both_0    = ((v1 == 0) & (v2 == 0)).sum()
    r1_only   = ((v1 == 1) & (v2 == 0)).sum()   # détecté seulement avec experts
    r2_only   = ((v1 == 0) & (v2 == 1)).sum()   # détecté seulement sans experts

    # Cohen's kappa (gère le cas où une colonne est constante)
    try:
        kappa = cohen_kappa_score(v1, v2)
    except Exception:
        kappa = np.nan

    stats_rows.append({
        "emotion":           e,
        "accord":            agree,
        "accord_pct":        round(agree_pct, 1),
        "kappa":             round(kappa, 3) if not np.isnan(kappa) else "N/A",
        "les_deux_1":        both_1,
        "les_deux_0":        both_0,
        "run1_seul":         r1_only,
        "run2_seul":         r2_only,
        "total_divergences": r1_only + r2_only,
    })

df_stats = pd.DataFrame(stats_rows)
print(df_stats.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
#  4. ACCORD GLOBAL (EXACT MATCH SUR LES 11 ÉMOTIONS)
# ════════════════════════════════════════════════════════════════════════════

exact_match = []
for _, row in merged.iterrows():
    match = all(row[f"{e}_r1"] == row[f"{e}_r2"] for e in EMOTIONS)
    exact_match.append(match)
merged["exact_match"] = exact_match

n_exact   = sum(exact_match)
pct_exact = n_exact / N * 100

print(f"\n{'='*70}")
print(f"  EXACT MATCH (11/11 émotions identiques)")
print(f"{'='*70}")
print(f"  {n_exact} / {N}  ({pct_exact:.1f}%)")

# Nombre d'émotions divergentes par message
n_diverg = []
for _, row in merged.iterrows():
    d = sum(1 for e in EMOTIONS if row[f"{e}_r1"] != row[f"{e}_r2"])
    n_diverg.append(d)
merged["n_divergences"] = n_diverg

print(f"\n  Distribution du nombre de divergences par message :")
print(merged["n_divergences"].value_counts().sort_index().to_string())

# ════════════════════════════════════════════════════════════════════════════
#  5. VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

# ── 5a. Accord (%) par émotion ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2ecc71" if v >= 90 else "#f39c12" if v >= 80 else "#e74c3c"
          for v in df_stats["accord_pct"]]
bars = ax.barh(
    df_stats["emotion"][::-1],
    df_stats["accord_pct"][::-1],
    color=colors[::-1], edgecolor="black", linewidth=0.5,
)
ax.axvline(90, color="grey", linestyle="--", linewidth=0.8, label="90%")
ax.set_xlabel("Accord (%)")
ax.set_title("Accord inter-runs par émotion (run1 vs run2)")
ax.set_xlim(0, 105)
for bar, val in zip(bars, df_stats["accord_pct"][::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "accord_par_emotion.png"), dpi=150)
plt.show()

# ── 5b. Divergences : qui détecte plus ? ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(EMOTIONS))
w = 0.35
ax.barh(x + w/2, df_stats["run1_seul"], w, label="Run1 seul (avec experts)", color="#3498db")
ax.barh(x - w/2, df_stats["run2_seul"], w, label="Run2 seul (sans experts)", color="#e67e22")
ax.set_yticks(x)
ax.set_yticklabels(df_stats["emotion"])
ax.set_xlabel("Nombre de messages")
ax.set_title("Divergences : quelle run détecte l'émotion que l'autre ne détecte pas ?")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "divergences_direction.png"), dpi=150)
plt.show()

# ── 5c. Histogramme nombre de divergences ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bins = range(0, merged["n_divergences"].max() + 2)
ax.hist(merged["n_divergences"], bins=bins, align="left",
        color="#9b59b6", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Nombre d'émotions divergentes")
ax.set_ylabel("Nombre de messages")
ax.set_title("Distribution du nombre de divergences par message")
ax.set_xticks(range(0, merged["n_divergences"].max() + 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_divergences.png"), dpi=150)
plt.show()

# ════════════════════════════════════════════════════════════════════════════
#  6. EXEMPLES DE DIVERGENCES (avec texte)
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  EXEMPLES DE MESSAGES AVEC DIVERGENCES")
print(f"{'='*70}")

df_diverg = merged[merged["n_divergences"] > 0].sort_values("n_divergences", ascending=False)

examples = []
for _, row in df_diverg.iterrows():
    idx = int(row["idx"])

    # Récupérer le texte original
    text = "N/A"
    name = "?"
    role = "?"
    if df_text is not None and idx < len(df_text):
        text = str(df_text.iloc[idx].get("TEXT", "N/A"))
        name = str(df_text.iloc[idx].get("NAME", "?"))
        role = str(df_text.iloc[idx].get("ROLE", "?"))

    # Lister les émotions divergentes
    diverg_details = []
    for e in EMOTIONS:
        v1 = int(row[f"{e}_r1"])
        v2 = int(row[f"{e}_r2"])
        if v1 != v2:
            direction = "run1=1 run2=0" if v1 == 1 else "run1=0 run2=1"
            diverg_details.append({"emotion": e, "run1": v1, "run2": v2, "direction": direction})

    examples.append({
        "idx":            idx,
        "row_id":         row.get("row_id_r1", idx),
        "name":           name,
        "role":           role,
        "text":           text,
        "n_divergences":  int(row["n_divergences"]),
        "divergences":    diverg_details,
        "rationale_r1":   row.get("rationale_r1"),
        "rationale_r2":   row.get("rationale_r2"),
        "confidence_r1":  row.get("confidence_r1"),
        "confidence_r2":  row.get("confidence_r2"),
    })

# Afficher les 15 premiers (les plus divergents)
for ex in examples[:15]:
    print(f"\n  ── idx={ex['idx']}  [{ex['name']}]  role={ex['role']}  "
          f"divergences={ex['n_divergences']} ──")
    print(f"  TEXT: \"{ex['text'][:120]}{'…' if len(ex['text']) > 120 else ''}\"")
    print(f"  Confiance : run1={ex['confidence_r1']}  run2={ex['confidence_r2']}")
    for d in ex["divergences"]:
        arrow = "→" if d["run1"] == 1 else "←"
        print(f"    {arrow} {d['emotion']:15s}  run1={d['run1']}  run2={d['run2']}")
    if ex["rationale_r1"]:
        print(f"  Rationale R1: {ex['rationale_r1']}")
    if ex["rationale_r2"]:
        print(f"  Rationale R2: {ex['rationale_r2']}")

# ════════════════════════════════════════════════════════════════════════════
#  7. EXPORT DES DIVERGENCES
# ════════════════════════════════════════════════════════════════════════════

# ── 7a. XLSX des divergences ───────────────────────────────────────────────
rows_export = []
for ex in examples:
    row_out = {
        "idx":           ex["idx"],
        "row_id":        ex["row_id"],
        "name":          ex["name"],
        "role":          ex["role"],
        "text":          ex["text"],
        "n_divergences": ex["n_divergences"],
        "confidence_r1": ex["confidence_r1"],
        "confidence_r2": ex["confidence_r2"],
        "rationale_r1":  ex["rationale_r1"],
        "rationale_r2":  ex["rationale_r2"],
    }
    # Colonnes par émotion : run1 / run2 / match
    for e in EMOTIONS:
        r1_val = int(merged.loc[merged["idx"] == ex["idx"], f"{e}_r1"].values[0])
        r2_val = int(merged.loc[merged["idx"] == ex["idx"], f"{e}_r2"].values[0])
        row_out[f"{e}_r1"] = r1_val
        row_out[f"{e}_r2"] = r2_val
        row_out[f"{e}_match"] = "✓" if r1_val == r2_val else "✗"
    rows_export.append(row_out)

df_export = pd.DataFrame(rows_export)

OUT_XLSX_COMP = os.path.join(OUT_DIR, "comparaison_divergences.xlsx")

with pd.ExcelWriter(OUT_XLSX_COMP, engine="openpyxl") as writer:
    # Onglet 1 : toutes les divergences
    df_export.to_excel(writer, sheet_name="divergences", index=False)

    # Onglet 2 : stats par émotion
    df_stats.to_excel(writer, sheet_name="accord_par_emotion", index=False)

    # Onglet 3 : tous les messages (avec flag match)
    cols_all = ["idx", "exact_match", "n_divergences"]
    for e in EMOTIONS:
        cols_all += [f"{e}_r1", f"{e}_r2"]
    merged[cols_all].to_excel(writer, sheet_name="tous_messages", index=False)

    # Onglet 4 : résumé
    summary = pd.DataFrame([{
        "messages_comparés":       N,
        "exact_match":             n_exact,
        "exact_match_pct":         round(pct_exact, 1),
        "messages_avec_divergence": len(df_diverg),
        "moy_divergences_par_msg": round(merged["n_divergences"].mean(), 2),
        "max_divergences":         int(merged["n_divergences"].max()),
        "émotion_plus_stable":     df_stats.loc[df_stats["accord_pct"].idxmax(), "emotion"],
        "émotion_moins_stable":    df_stats.loc[df_stats["accord_pct"].idxmin(), "emotion"],
    }])
    summary.T.to_excel(writer, sheet_name="resume", header=["valeur"])

print(f"\n✓ Comparaison exportée → {OUT_XLSX_COMP}")

# ════════════════════════════════════════════════════════════════════════════
#  8. RÉSUMÉ FINAL EN CONSOLE
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  RÉSUMÉ COMPARAISON RUN1 (avec experts) vs RUN2 (sans experts)")
print(f"{'='*70}")
print(f"  Messages comparés           : {N}")
print(f"  Exact match (11/11)         : {n_exact} ({pct_exact:.1f}%)")
print(f"  Messages avec ≥1 divergence : {len(df_diverg)} ({len(df_diverg)/N*100:.1f}%)")
print(f"  Moy. divergences/message    : {merged['n_divergences'].mean():.2f}")
print(f"  Émotion la + stable         : {df_stats.loc[df_stats['accord_pct'].idxmax(), 'emotion']} "
      f"({df_stats['accord_pct'].max():.1f}%)")
print(f"  Émotion la – stable         : {df_stats.loc[df_stats['accord_pct'].idxmin(), 'emotion']} "
      f"({df_stats['accord_pct'].min():.1f}%)")

# Kappas
print(f"\n  Cohen's Kappa par émotion :")
for _, r in df_stats.iterrows():
    bar = "█" * int(float(r["kappa"]) * 20) if r["kappa"] != "N/A" else "—"
    print(f"    {r['emotion']:15s}  κ = {str(r['kappa']):>6s}  {bar}")

print(f"\n✓ Figures dans       : {OUT_DIR}/")
print(f"✓ XLSX comparaison   : {OUT_XLSX_COMP}")
