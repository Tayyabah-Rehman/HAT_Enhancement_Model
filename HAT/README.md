[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-set5-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling?p=activating-more-pixels-in-image-super)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-urban100-4x)](https://paperswithcode.com/sota/image-super-resolution-on-urban100-4x?p=activating-more-pixels-in-image-super)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-set14-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling?p=activating-more-pixels-in-image-super)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/activating-more-pixels-in-image-super/image-super-resolution-on-manga109-4x)](https://paperswithcode.com/sota/image-super-resolution-on-manga109-4x?p=activating-more-pixels-in-image-super)

# HAT for License Plate Super-Resolution

**MPhil Thesis | Pakistan License Plate Enhancement | OCR Improvement**

<div align="center">
  <img src="figures/LR_Example.jpg" width="45%">
  <img src="figures/HR_Example.jpg" width="45%">
  <br>
  <em>Left: Low-Resolution Input | Right: HAT Enhanced Output (4× Upscale)</em>
</div>

## Setup

```bash
# Step 1: Python 3.10.11 venv
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip

# Step 2: Install PyTorch with GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Step 3: Clone and Install HAT
git clone https://github.com/XPixelGroup/HAT.git
cd HAT
pip install -r requirements.txt

# If basicsr fails, run these:
pip install wheel
pip install basicsr==1.3.4.9 --no-build-isolation --no-deps
pip install opencv-python lmdb yapf tb-nightly addict future pyyaml requests scikit-image scipy tqdm
pip install -r requirements.txt
pip install einops
python setup.py develop

# Step 4: Download weights to experiments/pretrained_models/
# Link: https://drive.google.com/file/d/1Ma12vCWT27P9M99-s2RXnynKN-OQsBrv/view

# Step 5: Fix errors
cd .venv\Lib\site-packages\basicsr\data
python -c "with open('degradations.py', 'r') as f: content = f.read(); content = content.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale'); open('degradations.py', 'w').write(content); print('Fixed!')"
cd ..\..\..\..\HAT
python -c "f=open('hat/archs/__init__.py','rb');raw=f.read();f.close();raw=raw.replace(b'\x00',b'');text=raw.decode('utf-8').strip();text=text+'\nfrom .hat_arch import HAT\n' if 'from .hat_arch import HAT' not in text else text;f=open('hat/archs/__init__.py','w',encoding='utf-8');f.write(text);f.close();print('Done')"
python -c "from hat.archs import HAT; print('HAT imported successfully!')"

# Step 6: Run enhancement
python enhance_hat_plates.py
