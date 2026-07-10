"""Compatibility wrapper for the two standalone competition figure scripts."""

from plot_competition_config_sensitivity import main as plot_config
from plot_competition_ood import main as plot_ood

if __name__ == "__main__":
    plot_ood()
    plot_config()
