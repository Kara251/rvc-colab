# rvc-colab

Automated RVC voice-model training on Google Colab via the official Colab CLI.
Trains one model per Blue Archive character (JP voices, all costumes merged),
resuming from Drive-synced checkpoints when sessions die.

## Layout

- `run_cloud.py` — local entry point; session lifecycle + resume loop
- `remote/worker.py` — runs on the Colab VM; Applio setup + training queue
- `remote/bootstrap.py`, `remote/poll.py` — exec snippets for the orchestrator
- `tools/gen_manifest.py` — scan VoiceData -> `models.yaml`
- `tools/prepare_datasets.py` — pack per-model JP ogg zips
- `tools/pull_models.py` — Drive exported -> local `RVC/` (auto `_v2` suffix)
- `models.yaml` — training queue manifest
- `config.yaml` — paths, Applio pin, defaults
- `vendor/Applio` — pinned upstream clone for reference (gitignored)

## Usage

```
python tools/gen_manifest.py        # regenerate models.yaml
python tools/prepare_datasets.py    # build dataset zips
python run_cloud.py                 # run the whole queue
python tools/pull_models.py         # fetch finished models
```

Drive layout: `RVC-Train/{logs,exported,cache,status.json}`.
See `docs/usage.md` for details.
