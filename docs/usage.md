# rvc-colab usage

## One-time setup

1. `uv tool install google-colab-cli`, then run any `colab` command and finish
   the browser OAuth flow once.
2. `python tools/gen_manifest.py` — scans `VoiceData/*/audios/jp_*/*.ogg`,
   writes `models.yaml`. Multi-speaker dirs (`Main`, `Tutorial`, `ST*`) are
   excluded; edit `models.yaml` to drop or retune models.
3. `python tools/prepare_datasets.py` — writes `datasets/<Model>.zip`
   (flat ogg bundles, stored uncompressed).

## Running

`python run_cloud.py` handles the whole queue:

- creates/reuses session `rvc-train` (T4 by default, `--gpu` overrides)
- uploads `worker.py` + config + model.json, pushes the local checkpoint
  mirror back to the VM, uploads the dataset zip, spawns the worker detached
- worker installs Applio (pinned commit) + prerequisites once per session,
  then preprocess -> extract -> train; Applio resumes automatically from the
  newest `G_*/D_*` checkpoint present in `logs/<model>/`
- orchestrator polls status and pulls new checkpoint files to
  `checkpoints/<model>/` every poll cycle, so progress survives VM death
- on `trained`, the latest `*_e_*s.pth` + `*.index` are copied into
  `RVC/<char>/<model>.pth|.index` (`_v2`, `_v3`... if the name is taken) and
  the remote logs dir is wiped to keep the VM disk clean

Interrupt with Ctrl-C; rerun the same command to continue. `--model NAME`
trains a single model, `--once` avoids the relaunch loop.

## Epochs

`gen_manifest.py` sets `epochs = clamp(target_steps * batch_size / files,
300, 1200)` — constant gradient steps (~25k) regardless of dataset size.
Override per model in `models.yaml`.

## Notes / limits

- Drive is NOT used: `colab drivemount` requires a fresh browser approval per
  session, which would break unattended resume. All persistence goes through
  `colab upload`/`download` over the session channel.
- preprocess/extract outputs are re-derived per session (minutes); only
  checkpoints, config.json and index files are mirrored.
- If a G_/D_ pair is corrupt (VM died mid-write), the worker deletes it and
  resumes from the previous pair.
- Failed models are logged and skipped; the queue continues.
- Dataset zips are uploaded on demand and deleted after unpacking; datasets
  and per-model preprocess artifacts never persist off the VM.
