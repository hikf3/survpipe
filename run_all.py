# surv2/run_all.py
from __future__ import annotations

from surv2.pipeline_survival import run_outcome
from surv2 import config as C
from surv2.utils_surv import log

def main():
    log(f"[RUN] DATA_PATH={C.DATA_PATH}")
    log(f"[RUN] OUTDIR={C.OUTDIR}")
    log(f"[RUN] outcomes={list(C.OUTCOMES.keys())}")
    for outcome in sorted(C.OUTCOMES.keys()):
        run_outcome(outcome)
    log("[RUN] main training+test pipeline complete.")
    log("[RUN] If you want KM plots with CI/logrank/at-risk: python -m surv2.posthoc_plots")

if __name__ == "__main__":
    main()
