from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    if importlib.util.find_spec('pandas'):
        run([sys.executable, 'dataset/merge_csv.py'])
    else:
        print('[warn] pandas missing: skipping merge_csv.py, using existing data/raw/all_runs.csv if present.')

    run([sys.executable, 'scripts/report_label_quality.py', '--csv', 'data/raw/all_runs.csv', '--h', '60'])
    run([sys.executable, 'scripts/evaluate_rule_baseline.py', '--csv', 'data/raw/all_runs.csv', '--h', '60', '--threshold', '0.5'])
    run([sys.executable, 'scripts/run_robustness_tests.py', '--csv', 'data/raw/all_runs.csv', '--threshold', '0.5'])


if __name__ == '__main__':
    main()
