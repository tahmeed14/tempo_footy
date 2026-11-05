"""Train soccer-tempo models."""

import arviz as az
import bambi as bmb

az.rcParams["stats.ci_kind"] = "hdi"
az.rcParams["stats.ci_prob"] = 0.95
