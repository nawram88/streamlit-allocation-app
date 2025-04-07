
import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Optimisation d'Allocation d'Actifs avec ROE et Rendement Assuré")

# Paramètres d'entrée
capital = st.number_input("Capital de départ (€)", value=100_000_000)
tax_rate = st.slider("Taux d'imposition (%)", 0, 100, 30) / 100
seuil_roe_actionnaire = st.slider("Seuil ROE actionnaire (%)", 0, 100, 10) / 100
seuil_rdt_assure = st.slider("Seuil rendement assuré (%)", 0, 10, 3.3) / 100
step = st.slider("Pas d'allocation (%)", 1, 20, 10)

# Hypothèses
asset_classes = ['Taux', 'Actions_cotees', 'Actions_non_cotees', 'Immobilier', 'Cash']
returns = {'Taux': 0.02, 'Actions_cotees': 0.03, 'Actions_non_cotees': 0.04, 'Immobilier': 0.035, 'Cash': 0.005}
capital_gains = {'Taux': 0.0, 'Actions_cotees': 0.04, 'Actions_non_cotees': 0.05, 'Immobilier': 0.03, 'Cash': 0.0}
volatility = {'Taux': 0.01, 'Actions_cotees': 0.15, 'Actions_non_cotees': 0.20, 'Immobilier': 0.10, 'Cash': 0.005}
correlation_matrix = np.array([
    [1.0, 0.2, 0.15, 0.1, 0.05],
    [0.2, 1.0, 0.6, 0.4, 0.05],
    [0.15, 0.6, 1.0, 0.5, 0.05],
    [0.1, 0.4, 0.5, 1.0, 0.05],
    [0.05, 0.05, 0.05, 0.05, 1.0]
])
vol_vector = np.array([volatility[a] for a in asset_classes])
cov_matrix = np.outer(vol_vector, vol_vector) * correlation_matrix

alloc_range = np.arange(0, 101, step)
allocations = [list(a) for a in product(alloc_range, repeat=5) if sum(a) == 100]

results = []
for alloc in allocations:
    weights = np.array([a/100 for a in alloc])
    revenu = sum(weights[i] * returns[asset_classes[i]] for i in range(5)) * capital
    gain = sum(weights[i] * capital_gains[asset_classes[i]] for i in range(5)) * capital
    revenu_net = revenu * (1 - tax_rate)
    roe = (revenu_net + gain) / capital
    risk = np.sqrt(weights.T @ cov_matrix @ weights)
    rdt_assure = sum(weights[i] * returns[asset_classes[i]] for i in [0, 4])
    results.append({**{asset_classes[i]: alloc[i] for i in range(5)}, 'ROE': roe, 'Risque': risk, 'Rendement_assure': rdt_assure})

df = pd.DataFrame(results)
df_valid = df[(df['ROE'] >= seuil_roe_actionnaire) & (df['Rendement_assure'] >= seuil_rdt_assure)]

st.subheader("Résultats")
st.dataframe(df_valid.sort_values(by="ROE", ascending=False).reset_index(drop=True))

# Graphe
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(df["Risque"], df["ROE"], alpha=0.2, label="Tous les portefeuilles")
ax.scatter(df_valid["Risque"], df_valid["ROE"], color='blue', alpha=0.7, label="Portefeuilles admissibles")
ax.axhline(seuil_roe_actionnaire, color='red', linestyle='--', label=f"Seuil ROE = {seuil_roe_actionnaire*100:.1f}%")
ax.axhline(seuil_rdt_assure, color='green', linestyle='--', label=f"Rdt assuré = {seuil_rdt_assure*100:.1f}%")
ax.set_xlabel("Risque")
ax.set_ylabel("ROE")
ax.set_title("ROE vs Risque de marché")
ax.legend()
st.pyplot(fig)

# Recommandation
if not df_valid.empty:
    best = df_valid.loc[df_valid['Risque'].idxmin()]
    st.success("✅ Recommandation optimale :")
    st.write(best)
else:
    st.error("❌ Aucun portefeuille admissible avec les contraintes actuelles.")
