# Kaggle submission and Moodle evidence

Team name: **General** (same members on both tracks). Check entry and team-merger deadlines on Kaggle.

## Code track (ARC Prize 2026 — ARC-AGI-3)
1. Set `configs/submission.yaml` to inherit the final agent. Budget there is `max_actions_per_game`.
2. If using pretrained weights, upload the `.pt` as a Kaggle dataset, add it to `dataset_sources` in
   `notebooks/kernel-metadata.json`, and set `init_checkpoint: /kaggle/input/<dataset>/final.pt`.
3. `make submit`, wait for the commit run, then **Submit to Competition** on the kernel page.
4. Internet is disabled in accelerated sessions; nothing in the agent needs it.

## Paper track
- Condensed write-up < 1500 words: `report/paper_track_writeup.md`. It must describe the linked code submission.

## Evidence for the Moodle ZIP (`General.zip`)
- [ ] Report PDF (RLC template, cover page disabled, ≤ 8 pages main content)
- [ ] Kaggle notebook (downloaded from the nominated submission), credentials removed
- [ ] Code submission identifier: ____
- [ ] Public score + timestamp screenshot: ____
- [ ] Nominated class-leaderboard entry: ____
- [ ] Paper Track write-up link: ____ and saved copy of submitted text
- [ ] Member contribution statement (`docs/CONTRIBUTIONS.md`)
