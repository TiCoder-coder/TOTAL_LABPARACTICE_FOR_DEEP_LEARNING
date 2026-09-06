# PHASE 31 CONDITION DISPATCH WEIGHT-DECAY KEY ISSUE

## 1. Observed failure

Phase 31 stopped before run registration with:

```text
KeyError: weight_decay
```

No Phase 31 run record, heartbeat or live result was created.

## 2. Root cause

`prepare_phase_31_condition` correctly returns the selected condition value in the top-level `weight_decay` field. Its frozen Phase 30 handoff contains only parameters that remain fixed and therefore does not contain `weight_decay`.

`run_single_condition.py` attempted to read `weight_decay` from `frozen_configuration` before applying the condition value. This violates the ownership boundary between frozen parameters and the currently swept factor.

## 3. Correct ownership

```text
Phase 31 weight_decay comes from phase_condition.weight_decay
Phase 32 dropout comes from phase_condition.dropout_probability
all non-swept parameters come from frozen_configuration
```

## 4. Safety result

```text
training started: false
registry record created: false
heartbeat created: false
live result created: false
upstream artifacts changed: false
```

