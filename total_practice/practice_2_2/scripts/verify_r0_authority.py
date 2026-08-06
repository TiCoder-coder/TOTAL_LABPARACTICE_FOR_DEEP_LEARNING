from __future__ import annotations

import argparse
import json
from pathlib import Path

from practice_2_2.r0_authority import build_r0_inventory, write_r0_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inventory = build_r0_inventory()
    if args.output is not None:
        write_r0_inventory(inventory, args.output)
    print(json.dumps(inventory, indent=2))
    if inventory["status"] != "ready_for_manual_authorization":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
