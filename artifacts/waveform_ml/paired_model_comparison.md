### Paired Patient-Clustered Bootstrap Comparison (Fold 10)

> [!NOTE]
> **Paired Bootstrap Design:** Evaluated across 2,198 Fold 10 records sharing 1,904 unique patient clusters. In each of 2,000 bootstrap iterations (seed: 42), identical patient clusters were resampled with replacement for both `Full-Fold Model` and `Pilot Model`. The paired difference $\Delta = \text{Macro F1}_A - \text{Macro F1}_B$ directly measures paired gain while controlling for intra-patient correlation.

**Macro F1 Point Delta:** +0.1510 | **95% Paired CI:** [0.1343, 0.1681] | **Bootstrap SE:** 0.0087 | **Outcome:** **Significant (+)**

| Superclass | Pilot Model F1 | Full-Fold Model F1 | Paired Delta ($\Delta$) | Bootstrap SE | 95% Paired CI | 95% Significant? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 0.5882 | **0.7668** | **+0.1785** | 0.0142 | [0.1503, 0.2071] | Yes (Positive) |
| **MI** | 0.3883 | **0.5594** | **+0.1711** | 0.0186 | [0.1356, 0.2070] | Yes (Positive) |
| **STTC** | 0.4909 | **0.6006** | **+0.1097** | 0.0156 | [0.0789, 0.1398] | Yes (Positive) |
| **CD** | 0.4933 | **0.6002** | **+0.1068** | 0.0176 | [0.0729, 0.1406] | Yes (Positive) |
| **HYP** | 0.3305 | **0.5195** | **+0.1890** | 0.0266 | [0.1346, 0.2393] | Yes (Positive) |
| **Macro Aggregate** | 0.4583 | **0.6093** | **+0.1510** | 0.0087 | **[0.1343, 0.1681]** | **Yes (Positive)** |