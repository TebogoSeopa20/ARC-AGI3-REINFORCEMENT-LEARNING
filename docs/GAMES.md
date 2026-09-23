# Environment disclosure

Every game touched during development is listed here (brief §4.3). Public set: 25 games, fetched 23 Sept 2026.
Levels come from `make games` (number of human-baseline entries).

| Game id | Version | Split | Levels | Used for | First used |
|---|---|---|---|---|---|
| cn04 | 2fe56bfb | dev | | smoke test (random/DQN/PPO, 300 actions) | 23 Sept 2026 |
| ka59 | | dev | | | |
| lp85 | | dev | | | |
| ls20 | | dev | | | |
| r11l | | dev | | | |
| re86 | | dev | | | |
| s5i5 | | dev | | | |
| sb26 | | dev | | | |
| sc25 | | dev | | | |
| sk48 | | dev | | | |
| sp80 | | dev | | | |
| su15 | | dev | | | |
| tn36 | | dev | | | |
| tu93 | | dev | | | |
| wa30 | | dev | | | |
| ar25 | | heldout | | final evaluation only | |
| bp35 | | heldout | | final evaluation only | |
| cd82 | | heldout | | final evaluation only | |
| dc22 | | heldout | | final evaluation only | |
| ft09 | | heldout | | final evaluation only | |
| g50t | | heldout | | final evaluation only | |
| lf52 | | heldout | | final evaluation only | |
| m0r0 | | heldout | | final evaluation only | |
| tr87 | | heldout | | final evaluation only | |
| vc33 | | heldout | | final evaluation only | |

Toy game (`src/arcrl/env/toy.py`): offline test harness for unit/smoke tests only. Never reported as a result.

Note: ls20, ft09 and vc33 appear in the official starter materials; ft09 and vc33 fell into held-out by the seeded split
and must not be used for development, including informal viewing of replays.
