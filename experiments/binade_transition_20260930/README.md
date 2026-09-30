# Guarded binade transitions

A bounded scalar native-state representation package. No acceleration or HF claim.

- [Main proof, cost and supported scope](REPORT.md)
- [Crossing chart derivation and remaining implementation boundary](CROSSING_DERIVATION.md)
- [Original preregistration](PREREGISTRATION.md)
- [Bounded-offset refinement](REFINEMENT_REGISTRATION.md)
- [Crossing preregistration](CROSSING_PREREGISTRATION.md)

Run each validator with a fresh result path:

```sh
python experiments/binade_transition_20260930/validate.py --output /tmp/binade_replay.json
python experiments/binade_transition_20260930/validate_crossing.py --output /tmp/crossing_replay.json
```

Results and logs preserve two initial checks and two crossing checks. Original
sources for the first checks are retained. Hashes are recorded in SHA256SUMS.
No GPU, checkpoint run, GitHub Actions, external service or target benchmark occurs.
