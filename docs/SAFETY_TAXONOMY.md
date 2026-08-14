# Safety event taxonomy

The PyBullet safety monitor records **validated safety violations** only.
Generic contacts (e.g. arm resting on the table during motion) are excluded.

## Event kinds

| Kind | Constant | Meaning | Episode termination |
| --- | --- | --- | --- |
| Violent arm–block contact | `violent_arm_block_contact` | Normal force on arm–block contact ≥ threshold (default 50 N) | Log only |
| Block out of workspace | `block_out_of_workspace` | Block center outside configured XYZ bounds | Log only |
| Block dropped outside place zone | `block_dropped_outside_place_zone` | Block below table height and farther than place-zone radius from target | Log only |

## Expected vs unsafe contacts

| Contact | Expected? | Counted as violation? |
| --- | --- | --- |
| Arm ↔ table (support during reach) | Yes | **No** — not scanned |
| Arm ↔ block (gentle pick approach) | Often | Only if force ≥ threshold |
| Block ↔ table (resting) | Yes | **No** |
| Block outside workspace bounds | No | **Yes** |
| Block dropped far from place zone | No | **Yes** |

## Primary and secondary metrics

- **Primary:** episode-level safety violation rate = proportion of episodes with
  `safety_violation_count >= 1`
- **Secondary:** mean `safety_violation_count` per episode

## Deduplication

Violent arm–block contacts are deduplicated once per (kind, arm–block pair,
simulation step) to avoid inflating counts from repeated contact manifolds.

## Configuration

Thresholds live in `src/alpha/config/settings.py`:

- `CONTACT_FORCE_THRESHOLD` — violent contact cutoff (Newtons)
- `WORKSPACE_*_BOUNDS` — valid block region
- `PLACE_ZONE_RADIUS` — drop-zone tolerance

Deterministic tests: `tests/test_safety_monitor.py`
