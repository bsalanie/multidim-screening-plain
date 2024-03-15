"""compare utility levels of the parties in different insurance models
with bidimensional types"""

import numpy as np
import pandas as pd

dim1 = dim2 = 20
type_grid = f"{dim1}x{dim2}"

models = ["d2_m1_deduc", "d2_m1_copay", "d2_m2"]
FB_surplus = np.zeros((dim1, dim2, 3))
SB_surplus = np.zeros((dim1, dim2, 3))
info_rents = np.zeros((dim1, dim2, 3))
not_insured = np.zeros((dim1, dim2, 3))

for imod, model in enumerate(models):
    df_all = pd.read_csv(
        f"results/insurance/{model}/insurance_{model}_{type_grid}/all_results.csv"
    )
    FB_surplus[:, :, imod] = df_all["FB_surplus"].values.reshape(dim1, dim2)
    SB_surplus[:, :, imod] = df_all["SB_surplus"].values.reshape(dim1, dim2)
    info_rents[:, :, imod] = df_all["info_rents"].values.reshape(dim1, dim2)

sigma = df_all["theta_0"].values.reshape(dim1, dim2)[0, :]
delta = df_all["theta_1"].values.reshape(dim1, dim2)[:, 0]

lost_surplus = FB_surplus - SB_surplus
SB_profits = SB_surplus - info_rents
EPS = 1e-6
not_insured = np.where(SB_surplus / FB_surplus < EPS, 1, 0)


print("\n The proportions of not insured in models 1, 2, and 3 are")
print(np.mean(not_insured, axis=(0, 1)))
print("\n")
