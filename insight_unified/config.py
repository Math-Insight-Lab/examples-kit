# -*- coding: utf-8 -*-
"""
Thin glue-config layer for examples-kit combined demos.
Only unify plot theme / export parameters, no modification to core-library source code.
MIT License
"""


class UnifiedConfig:
    """Shared configuration for all combined demos."""

    def __init__(self):
        self.theme = "teaching"
        self.export_fmt = ["png", "gif"]
        self.dpi = 150

    def set_theme(self, theme: str):
        self.theme = theme

    def set_export(self, fmt, dpi=150):
        self.export_fmt = fmt
        self.dpi = dpi

    def apply(self):
        """Apply unified config to both libraries."""
        # Apply theme to calc-insight-kit
        try:
            from calc_insight_kit import use_theme
            use_theme(self.theme)
        except ImportError:
            pass

        # Set matplotlib DPI
        import matplotlib.pyplot as plt
        plt.rcParams["figure.dpi"] = self.dpi


config = UnifiedConfig()
