from pathlib import Path

from dotenv import dotenv_values
from rich.console import Console
from rich.table import Table

from multidim_screening_plain.insurance_d2_m2_values import (
    cost_non_insur,
    expected_positive_loss,
    proba_claim,
)
from multidim_screening_plain.setup import setup_model

config = dotenv_values(
    Path.cwd() / "multidim_screening_plain" / "config_insurance_d2_m2.env"
)
model = setup_model(config)

cost_zero = cost_non_insur(model)
sigmas, deltas = model.theta_mat[:, 0], model.theta_mat[:, 1]
s = model.params[0]
accident_proba = proba_claim(deltas, s)
loss_pos = expected_positive_loss(deltas, s)

console = Console()

console.print("\n" + "-" * 80 + "\n", style="bold blue")

table = Table(title="Calibration for insurance_d2_m2")
table.add_column("Risk-aversion", justify="center", style="cyan", no_wrap=True)
table.add_column("Risk", justify="center", style="blue", no_wrap=True)
table.add_column("Proba claim", justify="center", style="green", no_wrap=True)
table.add_column("Exp. pos. loss", justify="center", style="red", no_wrap=True)
table.add_column("Cost non-insurance", justify="center", style="black", no_wrap=True)

for sigma, delta, cost, proba, loss in zip(
    sigmas, deltas, cost_zero, accident_proba, loss_pos, strict=True
):
    table.add_row(
        f"{sigma: > 8.3f}",
        f"{delta: > 8.3f}",
        f"{proba: > 8.3f}",
        f"{loss: > 8.3f}",
        f"{cost: > 8.3f}",
    )

console.print(table)
