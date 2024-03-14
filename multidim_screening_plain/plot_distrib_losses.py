"""plots the distribution of losses"""

import numpy as np
import scipy.stats as sts

n_delta_points = 9
n_loss_points = 100
delta_min, delta_max = 4.0, 8.0
s, k = 2.0, 0.012

deltas = np.linspace(delta_min, delta_max, num=n_delta_points)
losses = np.linspace(0.0, 15.0, num=n_loss_points)

pdf_losses = sts.norm.pdf(losses, loc=deltas, scale=s)
p_0 = k * deltas
proba_positive_losses = p_0 * sts.norm.cdf(deltas / s, loc=deltas, scale=s)
