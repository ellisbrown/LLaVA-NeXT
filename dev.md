# DEV


## Setup train conda env

```bash
conda create --prefix ./.conda/ov python=3.10 pip
conda activate ./.conda/ov
```

install conda system packages
```bash
# > from project root...
conda install -c conda-forge libaio -y

# clone git clone https://github.com/NVIDIA/cutlass.git into build/cutlass
mkdir -p build
git clone https://github.com/NVIDIA/cutlass.git build/cutlass


# set env vars for libaio
conda env config vars set CFLAGS="-I$CONDA_PREFIX/include"
conda env config vars set LDFLAGS="-L$CONDA_PREFIX/lib"
conda env config vars set CUTLASS_PATH="$PWD/build/cutlass"

# reactivate env
conda deactivate
conda activate ./.conda/ov
```



install pip packages
> NOTES:
>   - Must install `flash-attn` from source using the git url instead of the pip package.
>   - Must use an onld version of `accelerate` for compat with the pinned `transformers` commit.
```bash
pip install -e ".[train]"
pip install accelerate==0.29.1 pynvml
# install flash-attn from source https://github.com/Dao-AILab/flash-attention
pip install git+https://github.com/Dao-AILab/flash-attention.git
```
