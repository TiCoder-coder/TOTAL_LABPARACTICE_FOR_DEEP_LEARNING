from __future__ import annotations

import argparse
import json
from pathlib import Path

from practice_2_2.r1_contract import (
    build_r1_verification,
    load_r1_contract,
    write_json_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_r1_contract()
    pilot = json.loads(args.pilot.read_text())
    reviews = [json.loads(path.read_text()) for path in args.review]
    result = build_r1_verification(pilot, args.dataset_root, reviews, contract)
    write_json_record(result, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
