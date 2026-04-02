# Setup moi truong

## 1. Tao virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Cai dung PyTorch 2.9.1 voi CUDA 13.0

Bat buoc cai dung lenh sau:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

## 3. Cai cac package con lai

Sau khi cai PyTorch xong, cai tiep cac package con lai:

```bash
pip install -r requirements.txt
```

Neu da cai PyTorch bang lenh o buoc 2, lenh tren chi dong bo phien ban torch voi requirements va cai them cac package khac.

## 4. Kiem tra CUDA va GPU

Chay doan sau:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 5. Dieu kien de di tiep toi train

- `torch.__version__` phai la `2.9.1`
- `torch.version.cuda` phai hien `13.0`
- `torch.cuda.is_available()` phai la `True`
- Phai doc duoc ten GPU

Neu khong co GPU hoac CUDA khong dung, dung lai o buoc chuan bi va khong di tiep toi train.
