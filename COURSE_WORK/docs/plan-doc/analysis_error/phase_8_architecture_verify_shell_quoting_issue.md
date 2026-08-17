# Phase 8 Architecture Verify Shell Quoting Issue

## Issue ID

`CW-PHASE-8-ARCH-VERIFY-001`

## Status

`CONFIRMED`

## Failure

A read-only architecture search pattern contained Markdown backticks inside a double-quoted shell argument. The shell interpreted the enclosed file name as command substitution before running the search.

## Root cause

The verification command used an unsafe quoting form for literal backticks.

## Integrity impact

None. The attempted substitution referenced a non-executable path, no write command ran, and the architecture file remained unchanged.

## Required correction

Repeat verification with search patterns that omit backticks or use literal-safe quoting. Require all Phase 0-8 markers and reject all stale active Phase 0-7 markers.
