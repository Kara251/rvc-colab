# rvc-colab

Automated RVC voice-model training on Google Colab for Blue Archive JP voices
(all costumes merged per character). Google Drive holds everything: dataset
zips, checkpoints, finished exports. The notebook is the entry point — reopen
it and re-run the cell after a session dies and training resumes from Drive.

## Layout

- `worker.py` — runs inside the Colab session; installs Applio, loops
  `models.yaml`, syncs checkpoints to Drive every 60s, exports weights
- `notebook/train.ipynb` — the one-cell notebook you upload to Colab
- `tools/gen_manifest.py` — scan VoiceData -> `models.yaml`
- `tools/prepare_datasets.py` — pack per-model JP ogg zips
- `tools/pull_models.py` — collect exports into the local RVC library (`_v2` naming)
- `models.yaml` — training queue manifest
- `vendor/Applio` — pinned upstream clone for reference (gitignored)

## Usage

1. `python tools/gen_manifest.py` and `python tools/prepare_datasets.py` (done;
   outputs in `models.yaml` and `AI-Models/datasets/`)
2. On Drive create `RVC-Train/`; upload `datasets/` (all zips), `models.yaml`,
   and `worker.py` into it
3. Upload `notebook/train.ipynb` in Colab, switch runtime to T4 GPU, run the cell
4. Session dies -> reopen notebook, run the cell again. Done models are skipped.

Finished weights land in `RVC-Train/exported/<model>/`; download that folder,
then `python tools/pull_models.py --src <downloaded exported dir>` copies them
into `RVC/<char>/` with `_v2` suffixing on name clashes.
See `docs/usage.md` for details.
