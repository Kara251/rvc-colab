# rvc-colab usage

## One-time setup

1. `python tools/gen_manifest.py` — scans `VoiceData/*/audios/jp_*/*.ogg`,
   writes `models.yaml`. Multi-speaker dirs (`Main`, `Tutorial`, `ST*`) are
   excluded; edit `models.yaml` to drop or retune models.
2. `python tools/prepare_datasets.py` — writes `datasets/<Model>.zip`
   (flat ogg bundles, stored uncompressed).
3. On Google Drive create `RVC-Train/datasets/` and upload every zip from
   `AI-Models/datasets/` into it (~1.4 GB).
4. Push this repo to GitHub (`github.com/Kara251/rvc-colab`).

## Running

Open `notebook/train.ipynb` in Colab — upload it once, or use the GitHub URL
`colab.research.google.com/github/Kara251/rvc-colab/blob/main/notebook/train.ipynb`.
Pick a T4 GPU runtime, run the single cell. It mounts Drive (browser approval,
once per session), clones this repo, then runs `worker.py`, which:

- installs Applio (pinned commit) + pretrained weights once per session
- loops `models.yaml`: unzip dataset -> preprocess -> extract -> train
- syncs `logs/<model>/` (checkpoints, config, index) to
  `RVC-Train/logs/<model>/` every 60 s, so progress survives VM death
- on completion copies weights + index to `RVC-Train/exported/<model>/` and
  writes a `.trained` marker; done models are skipped on later runs

Session dies -> reopen the notebook, run the cell again. The worker restores
the newest Drive checkpoint and resumes; Applio picks up at the last saved
epoch automatically. Preprocess/extract outputs are re-derived per session
(minutes); only checkpoints and indexes persist on Drive.

## Collecting finished models

Download `RVC-Train/exported/` from Drive to local disk, then:

```
python tools/pull_models.py --src <path to exported>
```

copies each `<Model>.pth`/`.index` into `RVC/<char>/`, appending `_v2`, `_v3`…
when the name is already taken. Existing files are never modified.

## Epochs

`gen_manifest.py` sets `epochs = clamp(target_steps * batch_size / files,
300, 1200)` — roughly constant gradient steps (~25k) regardless of dataset
size. Override per model in `models.yaml` (`epochs`, `batch_size`,
`sample_rate`, `save_every_epoch`, `f0_method`, `embedder_model`).

## Notes / limits

- All state lives on Drive; your PC can be off between runs.
- One Drive approval click per session is required by Colab itself.
- A corrupt G_/D_ pair (VM died mid-write) is detected by Applio failing to
  load; delete the newest pair in `RVC-Train/logs/<model>/` to roll back.
- Failed models are logged and skipped; the queue continues.
- Dataset zips and preprocess artifacts stay on the VM and are deleted per
  model after export to keep the session disk clean.
