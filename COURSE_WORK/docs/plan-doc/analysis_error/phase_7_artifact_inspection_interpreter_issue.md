# Phase 7 Artifact Inspection Interpreter Issue

## Issue ID

`CW-PHASE-7-INSPECTION-INTERPRETER-001`

## Status

`CONFIRMED`

## Failure

Phase 7 materialization, sign-off, checksum checks and manifest inspection succeeded. A final read-only CSV inspection command used the system `python3`, which does not contain the project `pandas` dependency.

## Root cause

The inspection subprocess did not use the validated project interpreter.

## Integrity impact

None. The failure occurred after Phase 7 artifacts were signed and during a read-only inspection. No source, data or artifact was modified by the failing command.

## Required correction

Run the CSV inspection with `venv/bin/python`, require six rows in every per-variant audit and require every status to equal `PASS`.
