## Try the Examples

### Install packages for examples
```bash
pip install -r examples/requirements.txt
python -m pylo.util.patch_mup
```


```bash
cd examples
mkdir data
export PYLO_DATA_DIR=$PWD/data
pip install huggingface_hub
python download_dataset.py --output_dir data --repo_id btherien/imagenet-32x32x3
```

### Velo Example
```bash
python examples/mlp.py
```

### MuLO Example
```bash
python examples/mumlp.py
```