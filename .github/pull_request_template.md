## What changed

<!-- Brief summary of the changes -->

## Why

<!-- Motivation / linked issue -->

## Validation

- [ ] Tests pass locally (`python -m pytest tests/ -q`)
- [ ] CI passes (quality gate + tests + security)
- [ ] No secrets committed
- [ ] No raw/private telemetry committed
- [ ] No generated dataset committed
- [ ] No *.pkl/model artifact committed
- [ ] No user/session/device/event identifiers added to ML artifacts
- [ ] Documentation updated when behavior/contracts changed

### For ML changes

- [ ] Target semantics remain explicit
- [ ] No clinical claims introduced