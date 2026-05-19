# Face Swap Experiment Note

## What this experiment is about

This experiment checks whether a face-swapped image is:

1. far enough from the original target person's identity, and
2. close enough to the inserted reference identity,
3. and finally different enough that large-gallery face retrieval models no longer return the same nearest faces as for the original target.

In practical terms, the experiment tries to answer:

- does the swapped image still look like the original person for face-recognition models?
- does it move toward the reference identity?
- does this change hold not only in pairwise comparison, but also in gallery retrieval against FFHQ?


## Inputs

### Target images

- `test-img/female1.png` ... `female5.png`
- `test-img/male1.png` ... `male5.png`

These are the original people whose faces are replaced.

### Reference images

- `test-img/female_prot.png`
- `test-img/male_prot.png`

These are the source/reference identities inserted into the targets.

### External gallery

- `ffhq.tar.gz`

This archive is used in Stage 3 as a retrieval gallery. Only FFHQ images detected as frontal (`angle == 0` in FaceFusion) are included in the gallery index.


## Full experiment logic

The work is split into three stages.

### Stage 1. Face swapping

Goal:

- generate multiple swapped versions of each target image
- compare several swap models and parameter presets

Used models:

- `hyperswap_1a_256`
- `hyperswap_1c_256`
- `inswapper_128_fp16`
- `simswap_unofficial_512`

Used presets:

- `preset_A` - baseline
- `preset_B` - stronger identity replacement
- `preset_C` - smoother mask boundary
- `preset_D` - high-detail setting
- `preset_E` - swap + face enhancement

Important special case:

- `simswap_unofficial_512` could not complete with the nominal `preset_D` value `768x768`
- for this model the valid high-detail run is stored as `preset_D_1024x1024`

Outputs:

- 15 model/preset combinations
- 10 target images per combination
- 150 swapped result images total

Main script:

- `run_experiment.sh`


### Stage 2. Pairwise identity comparison

Goal:

- measure how much the swapped image moves away from the original target identity
- measure how much it moves toward the reference identity

Recognition models:

- ArcFace
- AdaFace

For each swapped image, embeddings are computed for:

- original target image
- reference image
- swapped image

Then cosine distances are computed:

- `original -> modified`
- `reference -> modified`

Interpretation:

- higher `original -> modified` distance means the target identity is less preserved
- lower `reference -> modified` distance means the swapped result is closer to the inserted reference identity

Main ranking metric:

- `identity_replacement_score = original_distance - reference_distance`

Higher score means better identity replacement.

Main script:

- `compute_face_metrics.py`

Main result of Stage 2:

- best combined setup: `hyperswap_1c_256 / preset_D`


### Stage 3. FFHQ retrieval experiment

Goal:

- test whether the embedding change is strong enough in a gallery-search setting
- not only compare image pairs, but also compare nearest-neighbor retrieval behavior

Why this stage matters:

- pairwise distance alone is not enough for the final claim
- a stronger test is whether the swapped image retrieves different nearest faces than the original image

Procedure:

1. build an FFHQ gallery index
2. keep only FFHQ faces with frontal orientation
3. compute ArcFace and AdaFace embeddings for gallery images
4. for each original target image, find its nearest FFHQ face
5. for each swapped image, retrieve top-5 nearest FFHQ faces
6. compare whether the swapped image still retrieves the same FFHQ identity that the original image retrieved

Frontal filtering:

- this is currently based on FaceFusion face analysis
- a gallery image is included when FaceFusion estimates `face.angle == 0`

Metrics used in Stage 3:

- `source_match_top1_rate`
- `source_match_top5_rate`
- `mean_source_gallery_distance`
- `mean_top1_ranker_distance`

Important interpretation:

- lower `source_match_top1_rate` and `source_match_top5_rate` are better
- if they go to `0.0`, the swapped image no longer retrieves the same FFHQ nearest identity as the original image

Main script:

- `compute_ffhq_retrieval.py`

Main result of Stage 3:

- FFHQ frontal gallery size: `4865`
- best setup for both ArcFace and AdaFace: `hyperswap_1c_256 / preset_D`
- `source_match_top1_rate = 0.0`
- `source_match_top5_rate = 0.0`


## Directory structure and what each part means

### Core scripts

- `run_experiment.sh`  
  Generates swapped images for the full model/preset matrix.

- `compute_face_metrics.py`  
  Computes ArcFace/AdaFace pairwise embedding distances for Stage 2.

- `compute_ffhq_retrieval.py`  
  Builds FFHQ frontal-face gallery index and computes retrieval metrics for Stage 3.


### Main notebook

Use your manually maintained Jupyter notebook as the main human-readable report.

Recommended chapter structure:

1. face swapping results
2. cosine-comparison results
3. FFHQ retrieval results


### Swapped image outputs

- `processed_data/<model>/<preset>/female/femaleN.png`
- `processed_data/<model>/<preset>/male/maleN.png`

Meaning:

- `<model>` = face-swapper model
- `<preset>` = parameter setup used for that run
- `female/` and `male/` = target-image group
- `femaleN.png` or `maleN.png` = exact target image index

Example:

- `processed_data/hyperswap_1c_256/preset_D/female/female3.png`

means:

- model `hyperswap_1c_256`
- preset `D`
- target group `female`
- target image `female3.png`


### Logs

- `logs/<model>/<preset>/<image>.log`

Each log corresponds to one generated swapped output.

Use logs when:

- a single case failed
- a model download stalled
- you need the exact FaceFusion command behavior for one output


### Stage 2 reports

- `reports/face_similarity_results.csv`
- `reports/face_similarity_summary.csv`
- `reports/face_similarity_combined_summary.csv`

Meaning:

- `face_similarity_results.csv`  
  one row per swapped image per recognizer

- `face_similarity_summary.csv`  
  summary per model/preset per recognizer

- `face_similarity_combined_summary.csv`  
  combined ArcFace + AdaFace ranking of setups


### Stage 3 reports

- `reports/ffhq_gallery_manifest.csv`
- `reports/ffhq_gallery_embeddings.npz`
- `reports/ffhq_retrieval_results.csv`
- `reports/ffhq_retrieval_query_summary.csv`
- `reports/ffhq_retrieval_experiment_summary.csv`

Meaning:

- `ffhq_gallery_manifest.csv`  
  list of FFHQ gallery images included in the frontal-face index

- `ffhq_gallery_embeddings.npz`  
  saved gallery embeddings for ArcFace and AdaFace

- `ffhq_retrieval_results.csv`  
  detailed top-5 nearest-neighbor results for every swapped image and each recognizer

- `ffhq_retrieval_query_summary.csv`  
  per-query summary, including whether the original target's gallery identity still appears in top-1 or top-5

- `ffhq_retrieval_experiment_summary.csv`  
  summary per model/preset per recognizer


### Notebook assets

- `reports/notebook_assets/`

Contains prebuilt PNG contact sheets used by the notebook.

These include:

- source-image overview
- per-experiment swapped-image grids
- FFHQ retrieval grids for Chapter 3
- extracted FFHQ gallery face images used inside retrieval summaries


## What the processed data means at article level

### What to say from Stage 1

You can describe:

- which swapping models were tested
- which presets were used
- that each setup was applied to the same 10 targets
- that the output matrix contains 150 swapped images

You should visually show:

- original targets
- reference faces
- representative swapped outputs for each model/preset


### What to say from Stage 2

You can claim:

- the swapped image becomes less similar to the original target in ArcFace/AdaFace embedding space
- the swapped image becomes relatively closer to the reference identity
- among tested setups, `hyperswap_1c_256 / preset_D` gives the strongest identity replacement score

Good material for the article:

- the combined ranking table
- a short explanation of cosine distance
- one or two representative examples


### What to say from Stage 3

This is the stronger result for the final article.

You can claim:

- the identity shift is strong enough to affect nearest-neighbor retrieval in a large gallery
- for the best setup, the swapped image no longer retrieves the same FFHQ nearest identity as the original target
- this held for both ArcFace and AdaFace in the current experiment

Good material for the article:

- the retrieval-summary table
- the top-5 retrieval contact sheets
- explicit mention that the best setup achieved `0.0` for both `source_match_top1_rate` and `source_match_top5_rate`


## Recommended article structure

One clean way to write the final article:

1. Problem statement  
   Explain why perceptual visual quality alone is not enough and why recognition-model behavior must be checked.

2. Experimental setup  
   Describe targets, reference faces, swap models, presets, ArcFace, AdaFace, and FFHQ gallery usage.

3. Face swapping results  
   Show qualitative examples from Chapter 1.

4. Pairwise embedding comparison  
   Present Stage 2 tables and explain `original -> modified` versus `reference -> modified`.

5. Retrieval experiment on FFHQ  
   Present Stage 3 as stronger evidence than pairwise distance alone.

6. Best-performing setup  
   State that `hyperswap_1c_256 / preset_D` is the strongest setup in the current experiment.

7. Limitations  
   Mention that frontal-face filtering currently uses FaceFusion `angle == 0`, and AdaFace ran on CPU in the current environment.


## Quick reminder for future work

If you come back later and forget what is important:

- `processed_data/` = actual swapped images
- `reports/face_similarity_*.csv` = Stage 2 numeric evidence
- `reports/ffhq_retrieval_*.csv` = Stage 3 numeric evidence
- `reports/notebook_assets/` = images used inside the notebook
- your main notebook = final combined report

If you need to explain the best result in one sentence:

- `hyperswap_1c_256 / preset_D` gave the strongest identity replacement in pairwise ArcFace/AdaFace comparison and also broke original-identity nearest-neighbor retrieval in the FFHQ gallery for both recognizers.
