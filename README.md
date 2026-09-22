# rvc-colab

Automated RVC voice-model training on Google Colab via the official Colab CLI.
Trains one model per Blue Archive character (JP voices, all costumes merged).
The orchestrator mirrors checkpoints to local disk, so a dead session just
gets recreated and the model resumes — no Google Drive needed.

## Layout

- `run_cloud.py` — local entry point; session lifecycle + queue + transfers
- `remote/worker.py` — runs on the Colab VM; Applio setup + one model per run
- `remote/bootstrap.py`, `remote/poll.py` — exec snippets for the orchestrator
- `tools/gen_manifest.py` — scan VoiceData -> `models.yaml`
- `tools/prepare_datasets.py` — pack per-model JP ogg zips
- `models.yaml` — training queue manifest
- `config.yaml` — paths, Applio pin, defaults
- `vendor/Applio` — pinned upstream clone for reference (gitignored)

## Usage

```
python tools/gen_manifest.py        # regenerate models.yaml
python tools/prepare_datasets.py    # build dataset zips
python run_cloud.py                 # run the whole queue
python run_cloud.py --model X_JP    # one model only
```

Checkpoints mirror to `checkpoints/<model>/`; finished weights+index land in
`RVC/<char>/<model>.pth|.index` (`_v2`, `_v3`... suffix when taken).
See `docs/usage.md` for details.
