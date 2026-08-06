from __future__ import annotations

import argparse
import json
from pathlib import Path

from practice_2_2.r1_contract import (
    build_pilot_manifest,
    build_review_template,
    load_r1_contract,
    write_json_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-template", type=Path)
    args = parser.parse_args()
    contract = load_r1_contract()
    pilot = build_pilot_manifest(args.dataset_root, contract)
    write_json_record(pilot, args.output)
    if args.review_template is not None:
        write_json_record(
            build_review_template(pilot, contract), args.review_template
        )
    print(json.dumps(pilot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
