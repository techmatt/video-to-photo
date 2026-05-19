# still-extractor

A local CLI pipeline that mines family videos for high-quality still frames worth keeping as photos. It indexes frames from source videos, scores them for sharpness, exposure, and aesthetic appeal, refines candidates with face- and identity-aware clustering, and produces a browsable HTML gallery of the best stills.

## Setup

This project targets Windows with an NVIDIA GPU and Python 3.12+.

1. Install [uv](https://docs.astral.sh/uv/) if you don't already have it.
2. Run `uv sync` — this installs everything, including the CUDA build of PyTorch. The `[tool.uv.sources]` block in `pyproject.toml` routes torch/torchvision/torchaudio to the cu124 wheels (compatible with NVIDIA 12.x drivers).

## Usage

The pipeline runs as five sequential scripts under `still_extractor/`:

- `still_extractor/pass1_index.py` — walk source videos and index candidate frames.
- `still_extractor/pass2_score.py` — score indexed frames for sharpness, exposure, and aesthetics.
- `still_extractor/pass3_refine.py` — refine top candidates with face/identity signals.
- `still_extractor/cluster_faces.py` — cluster detected faces into identities.
- `still_extractor/build_index_html.py` — emit a browsable HTML gallery of selected stills.

Run any of them as a module, e.g. `uv run python -m still_extractor.pass1_index --help`.

## Notes

InsightFace `buffalo_l` weights are licensed for non-commercial research use only. This project is intended for personal/family use.
