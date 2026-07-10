# Dataset Handoff Requirements

Send the dataset after the engine, API, and validation tooling are ready. That is the current stage.

## Minimum Dataset Level

- Legally usable and consent-cleared images only.
- One folder per identity or a manifest mapping every image to an identity.
- At least 2 images per identity for verification testing.
- Prefer 5-20 images per identity for enrollment/search development.
- Images should be face-visible, non-cartoon, non-synthetic unless explicitly marked.
- Minimum face crop target: 80 x 80 pixels or better.
- Avoid severe blur, masks, heavy occlusion, extreme profile, and very dark images for the first dataset.

## Required Manifest

Use CSV or JSON with these fields:

- `image_path`
- `identity_id`
- `workspace`
- `consent`
- `lawful_basis`
- `split`

Create a template:

```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/dataset_cli.py template dataset_manifest.csv
```

Validate before training:

```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/dataset_cli.py validate --root C:\path\to\dataset --manifest C:\path\to\dataset_manifest.csv --report runtime\dataset_report.json
```

## Recommended First Delivery

Start with a pilot dataset:

- 50-200 identities
- 5-10 images per identity
- Separate train/test split in the manifest
- Clear consent/lawful basis for every record

After the pilot validates cleanly, send the larger dataset.

## RecordIO / WebFace Zip Drops

Large face datasets may arrive as MXNet RecordIO archives, commonly containing `train.rec`, `train.idx`, `train.lst`, and benchmark `.bin` files. Catalog those archives first without extracting them:

```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/dataset_cli.py catalog-recordio-zip --source C:\path\to\faces_webface_112x112.zip --output-dir runtime\datasets --workspace webface --lawful-basis dataset_provider_terms
```

The generated manifest intentionally keeps `consent=false` unless `--consent` is passed. Do not train from a cataloged dataset until legal usage rights and consent status are confirmed.

## Image Archive Detection And Verification Datasets

Datasets such as WIDER Face validation archives contain scene images for detector/quality validation, not identity labels. Other archives such as CACD_VS may be used for verification validation. Catalog them separately:

```powershell
cd backend
$env:PYTHONPATH = (Get-Location).Path
python scripts/dataset_cli.py catalog-image-zip --source C:\path\to\WIDER_val.zip --output-dir runtime\datasets --workspace wider --task detection_validation --lawful-basis legal --consent
python scripts/dataset_cli.py catalog-image-zip --source C:\path\to\CACD_VS.tar --output-dir runtime\datasets --workspace cacd_vs --task verification_validation --lawful-basis legal --consent
```
