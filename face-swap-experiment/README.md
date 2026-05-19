# Face Swap Experiment

Inputs:

- `test-img/female1.png` through `female5.png`
- `test-img/male1.png` through `male5.png`
- Female source face: `test-img/female_prot.png`
- Male source face: `test-img/male_prot.png`

Outputs:

- `processed_data/<model>/<preset>/female/femaleN.png`
- `processed_data/<model>/<preset>/male/maleN.png`
- Logs: `logs/<model>/<preset>/<image>.log`
- Similarity measurements: `reports/face_similarity_results.csv`
- Per-recognizer summary: `reports/face_similarity_summary.csv`
- Combined summary: `reports/face_similarity_combined_summary.csv`

Completed matrix:

- `hyperswap_1a_256`: `preset_A`, `preset_B`, `preset_C`, `preset_D`, `preset_E`
- `hyperswap_1c_256`: `preset_A`, `preset_B`, `preset_D`
- `inswapper_128_fp16`: `preset_A`, `preset_B`, `preset_C`, `preset_E`
- `simswap_unofficial_512`: `preset_A`, `preset_B`, `preset_D_1024x1024`

Note:

`simswap_unofficial_512` failed with FaceFusion's advertised `768x768` pixel boost because the model is native `512x512`, and `768` cannot be split into an integer tile grid for this code path. The valid high-detail SimSwap run is stored as `preset_D_1024x1024`.

Recognition comparison:

The notebook compares each modified image against the original target image and the gender-matched reference image using ArcFace and AdaFace embeddings. The best combined setup by `original_distance - reference_distance` is `hyperswap_1c_256 / preset_D`.

Reproducible environment:

- From the repository root, run `nix develop`
- The flake creates and maintains a root `.venv` from `facefusion/requirements.txt` and `face-swap-experiment/requirements.txt`
