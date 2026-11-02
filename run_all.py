"""Regenerate every figure and table in figures/ (list in README.md).

Run from the project root:  python run_all.py
"""
import time

from studies import (bar1d, bending_error, element_checks, exact_field, midspan_stress,
                     one_element, reactions, tip_deflection)

STUDIES = [bar1d, element_checks, exact_field, bending_error, one_element,
           tip_deflection, midspan_stress, reactions]

if __name__ == "__main__":
    start = time.perf_counter()
    for study in STUDIES:
        t0 = time.perf_counter()
        print(f"===== {study.__name__} " + "=" * 60)
        study.main()
        print(f"({time.perf_counter() - t0:.1f} s)\n")
    print(f"All studies done in {time.perf_counter() - start:.1f} s; outputs in figures/")
