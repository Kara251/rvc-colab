# rvc-colab usage

## One-time setup

1. `uv tool install google-colab-cli`, then run any `colab` command and finish
   the browser OAuth flow once.
2. `python tools/gen_manifest.py` — scans `VoiceData/*/audios/jp_*/*.ogg`,
   writes `models.yaml`. Multi-speaker dirs (`Main`, `Tutorial`, `ST*`) are
   excluded; edit the file to drop or retune models.
3. `python tools/prepare_datasets.py` — writes `datasets/<Model>.zip`
   (flat ogg bundles, stored uncompressed).

## Running

`python run_cloud.py` does everything:

- creates (or reuses) session `rvc-train`, mounts Drive
- uploads `worker.py` + manifest, spawns the worker detached
- worker installs Applio (pinned commit) and downloads prerequisites; heavy
  assets are cached to Drive so later sessions skip the download
- per model: waits for its dataset zip (orchestrator uploads on demand),
  preprocess -> extract -> train; checkpoints sync to Drive every 60 s
- if the session dies (quota/idle), the loop recreates it and the worker
  resumes from the newest Drive checkpoint; preprocess/extract are re-run
  because their outputs are cheap and local-only
- finished models export `*_e_*s.pth` + `*.index` to `Drive/RVC-Train/exported/`

Interrupt with Ctrl-C; rerun the same command to continue. `--once` performs a
single session attempt without the relaunch loop.

## Epochs

`gen_manifest.py` sets `epochs = clamp(target_steps * batch_size / files,
300, 1200)` — constant gradient steps (~25k) regardless of dataset size.
Override per model in `models.yaml`.

## Pulling results

While a session is alive: `python tools/pull_models.py` copies each export into
`RVC/<Char>/<Char>_JP.pth|.index`, suffixing `_v2`, `_v3`, ... when the name is
taken.

## Notes / limits

- Checkpoints are the only large artifacts synced to Drive; preprocess slices
  and features are re-derived per session (minutes).
- If a G_/D_ pair is corrupt (died mid-write), the worker deletes it and
  resumes from the previous pair.
- Failed models are logged in `status.json` and skipped; the queue continues.
