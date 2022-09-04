# Just playing with the diffusers library from huggingface

## Setup virtual environment

```
python3.8 -m venv .venv
source ./.venv/bin/activate
```

## Installing PyTorch in a fresh venv, cuda v11.7

```
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu117
```

## Get it going

[Stable-Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

```
pip install --upgrade diffusers transformers scipy
huggingface-cli login
```