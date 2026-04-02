# YOLO lesion localization prep project

Project nay duoc tao de CHUAN BI train YOLO cho bai toan khoanh vung la bi ton thuong do benh. Pipeline uu tien YOLO segmentation khi label co polygon/mask, va tu dong fallback ve YOLO detection khi dataset chi co bounding box.

## Nguyen tac quan trong

**Chua train ngay. Chi chuan bi toi buoc train.**

- Khong tu dong goi training.
- Khong tu dong download model lon.
- Mac dinh uu tien model nano de tranh OOM tren RTX 5060 8GB VRAM.
- Cau hinh mac dinh an toan cho RAM 16GB.

## Yeu cau moi truong

- Python 3.10+
- CUDA 13.0
- PyTorch:
  - `torch==2.9.1`
  - `torchvision==0.24.1`
  - `torchaudio==2.9.1`
- Ultralytics YOLO

## Cau truc thu muc

```text
project_root/
  configs/
    thresholds.yaml
    yolo_train.yaml
  data/
    raw/
    raw_labels/
    cleaned/
    deduped/
    synthesized/
      images/
      labels/
    yolo/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      dataset.yaml
  docs/
    pipeline.md
  outputs/
    logs/
    quarantine/
    reports/
  scripts/
    setup_env.md
  src/
    utils/
    data_audit.py
    preprocess_images.py
    deduplicate_images.py
    synthesize_images.py
    augment_yolo_dataset.py
    validate_annotations.py
    build_yolo_dataset.py
    select_yolo_config.py
    train_yolo.py
  tests/
    smoke_test.py
  .gitignore
  README.md
  requirements.txt
```

## Dataset public da san sang train

Repo public nay da kem san dataset detection theo dung layout YOLO:

- `khoanhvungla/images/train`
- `khoanhvungla/images/val`
- `khoanhvungla/labels/train`
- `khoanhvungla/labels/val`
- `khoanhvungla/dataset.yaml`

`configs/yolo_train.yaml` mac dinh da tro thang toi `khoanhvungla/dataset.yaml`. Vi vay tren may khac, sau khi `git clone` + setup moi truong dung, ban co the train thu cong ngay ma khong can rebuild `data/yolo`.

## Chuan bi du lieu de chay lai pipeline

Dat anh goc vao `data/raw` va label goc vao `data/raw_labels`.

Neu ban dang co du lieu cu trong:

- `khoanhvungla/images`
- `khoanhvungla/labels`

thi hay copy vao:

- `data/raw`
- `data/raw_labels`

Pipeline hien tai khong tu dong di chuyen du lieu cu de tranh ghi de ngoai y muon. Muc nay chi can khi ban muon audit, clean, dedup, synthesize va build lai dataset theo pipeline.

## Setup moi truong

### 1. Tao virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Cai dung PyTorch 2.9.1 + CUDA 13.0

Bat buoc dung dung lenh sau:

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

### 3. Cai cac package con lai

```bash
pip install -r requirements.txt
```

README va `requirements.txt` giu nguyen bo torch theo dung yeu cau, nhung PyTorch van phai duoc cai bang `--index-url https://download.pytorch.org/whl/cu130`.

### 4. Kiem tra GPU truoc khi nghi den train

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Neu `torch.cuda.is_available()` la `False` thi dung lai o buoc chuan bi, khong di tiep toi train.

## Thu tu pipeline

1. Audit du lieu
2. Preprocess anh
3. Loai duplicate va near-duplicate
4. Sinh them anh cho class thieu mau
5. Cau hinh augmentation cho train
6. Validate annotation
7. Build dataset YOLO
8. Sinh config train an toan
9. In lenh train mau
10. Dung lai

## Quick start tren may moi

1. `git clone` repo
2. Tao `venv`
3. Cai PyTorch dung lenh CUDA 13.0
4. `pip install -r requirements.txt`
5. Kiem tra GPU
6. Chay lenh train thu cong khi ban da san sang

Lenh wrapper da tro san vao dataset public trong repo:

```bash
python src/train_yolo.py --config configs/yolo_train.yaml
```

Lenh Ultralytics CLI tuong duong:

```bash
yolo detect train data=khoanhvungla/dataset.yaml model=yolo11n imgsz=640 batch=8 epochs=100 workers=2 device=0 project=outputs/yolo name=lesion_localization
```

## Cac lenh CLI mau cho pipeline rebuild dataset

### 1. Setup moi truong

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### 2. Audit du lieu

```bash
python src/data_audit.py --input data/raw --output outputs
```

### 3. Preprocess anh

```bash
python src/preprocess_images.py --input data/raw --output data/cleaned --config configs/thresholds.yaml
```

### 4. Deduplicate

```bash
python src/deduplicate_images.py --input data/cleaned --mode report-only --output outputs
python src/deduplicate_images.py --input data/cleaned --mode apply --output data/deduped
```

### 5. Synthesize anh

```bash
python src/synthesize_images.py --input data/deduped --output data/synthesized --config configs/thresholds.yaml
```

### 6. Validate annotations

```bash
python src/validate_annotations.py --images data/synthesized --labels data/raw_labels --task detect
```

Neu dataset la segmentation, doi `--task segment`.

### 7. Build YOLO dataset

```bash
python src/build_yolo_dataset.py --images data/synthesized --labels data/raw_labels --output data/yolo
```

### 8. Select config train

```bash
python src/select_yolo_config.py --gpu_vram_gb 8 --ram_gb 16 --task detect --output configs/yolo_train.yaml
```

Neu dataset la segmentation, doi `--task segment`.

### 9. In lenh train mau

Lenh train bang wrapper Python:

```bash
python src/train_yolo.py --config configs/yolo_train.yaml
```

Lenh train Ultralytics CLI tham khao cho detection:

```bash
yolo detect train data=khoanhvungla/dataset.yaml model=yolo11n imgsz=640 batch=8 epochs=100 workers=2 device=0 project=outputs/yolo name=lesion_localization
```

Lenh train Ultralytics CLI tham khao cho segmentation:

```bash
yolo segment train data=data/yolo/dataset.yaml model=yolo11n-seg imgsz=640 batch=8 epochs=100 workers=2 device=0 project=outputs/yolo name=lesion_localization
```

Neu `select_yolo_config.py` chon family `yolo26`, thay model thanh `yolo26n` hoac `yolo26n-seg`.

### 10. Dung

Dung lai o day. Khong chay train trong buoc chuan bi.

## Giai thich tung script

### `src/data_audit.py`

- Quet `data/raw`
- Bao cao tong anh, anh loi, anh qua nho
- Thong ke kich thuoc min/max/avg
- Uoc luong ti le anh mo
- Thong ke class distribution neu co label
- Thong ke source distribution theo thu muc con
- Xuat:
  - `outputs/audit_report.json`
  - `outputs/audit_report.csv`

### `src/preprocess_images.py`

- Doc anh an toan
- Sua EXIF orientation
- Chuan hoa extension
- Loai anh loi, qua nho, qua mo, qua toi, qua chay
- Giu ti le anh
- Cho phep resize canh dai de giam tai bo nho ma khong pha YOLO label da normalize
- Ghi log:
  - `outputs/reports/preprocess_manifest.json`
  - `outputs/reports/preprocess_manifest.csv`
  - `outputs/reports/preprocess_rejected.json`
  - `outputs/reports/preprocess_rejected.csv`

### `src/deduplicate_images.py`

- Duplicate chinh xac bang `sha256`
- Near-duplicate bang `imagehash.phash`
- Che do:
  - `report-only`
  - `apply`
- Khi `apply`:
  - giu anh co do phan giai cao hon
  - uu tien anh co annotation quality tot hon
  - dua anh bi loai vao `quarantine/`, khong xoa vinh vien
- Xuat:
  - `outputs/duplicate_report.json`
  - `outputs/duplicate_report.csv`

### `src/synthesize_images.py`

- Khong dung API ngoai
- Copy toan bo anh goc vao `data/synthesized/images`
- Copy label tuong ung vao `data/synthesized/labels`
- Chi sinh them mau nhe cho class thieu du lieu
- Neu `segment`:
  - co the dung polygon de tao foreground-aware blend nhe
- Neu `detect`:
  - uu tien bbox-focused blend nhe
- Ghi metadata:
  - `is_synthetic`
  - `parent_image`
  - `method`
- Xuat:
  - `data/synthesized/synthesis_manifest.json`
  - `data/synthesized/synthesis_manifest.csv`

### `src/augment_yolo_dataset.py`

- Day la script augmentation offline tuy chon
- Chi augment split `train`
- Khong augment `val/test`
- Dung Albumentations
- Xac suat tung phep augment nam trong `configs/thresholds.yaml`
- Phu hop khi ban muon preview hoac frozen offline augmentation truoc train

### `src/validate_annotations.py`

- Kiem tra detect:
  - format YOLO bbox
  - class id hop le
  - bbox nam trong khung anh
- Kiem tra segment:
  - polygon hop le
  - du so diem
  - khong vuot khung anh
- Xuat:
  - `outputs/annotation_validation_report.json`
  - `outputs/annotation_validation_report.csv`

### `src/build_yolo_dataset.py`

- Input:
  - anh da preprocess/dedup/synth
  - label hop le
  - duplicate report neu co
  - synthesis manifest neu co
- Output:
  - `data/yolo/images/train`
  - `data/yolo/images/val`
  - `data/yolo/images/test`
  - `data/yolo/labels/train`
  - `data/yolo/labels/val`
  - `data/yolo/labels/test`
  - `data/yolo/dataset.yaml`
- Co gang stratified theo class
- Tranh de duplicate va synthetic parent bi tach sang nhieu split

### `src/select_yolo_config.py`

- Mac dinh uu tien nano:
  - `yolo11n`
  - `yolo11n-seg`
- Neu moi truong Ultralytics moi hon va thoa dieu kien tu dong, co the chon:
  - `yolo26n`
  - `yolo26n-seg`
- Mac dinh de xuat:
  - `imgsz=640`
  - `batch=8`
  - `workers=2`
  - `amp=true`
  - `cache=false`
  - `device=0`
- Neu du doan de OOM:
  - `batch: 8 -> 6 -> 4 -> 2`
  - neu can thi `imgsz: 640 -> 512`

### `src/train_yolo.py`

- La script train hoan chinh nhung **khong tu chay**
- Ho tro ca detect va segment thong qua config/model
- Co argparse:
  - `--config`
  - `--data`
  - `--model`
  - `--imgsz`
  - `--batch`
  - `--epochs`
  - `--workers`
  - `--device`
  - `--resume`
- Co bat loi CUDA OOM va goi yiam batch/image size

## Config quan trong

### `configs/thresholds.yaml`

Chua:

- blur threshold
- min image size
- brightness thresholds
- duplicate threshold
- synthesis target per class
- augmentation probabilities

### `configs/yolo_train.yaml`

Mac dinh:

- task detect hoac segment
- `data: khoanhvungla/dataset.yaml`
- model nano
- `imgsz: 640`
- `batch: 8`
- `workers: 2`
- `epochs: 100`
- `amp: true`
- `cache: false`
- `patience: 20`
- `device: 0`
- `project: outputs/yolo`
- `name: lesion_localization`

## Smoke test nhe

Khong train, chi test nhanh utility va config selection:

```bash
python tests/smoke_test.py
```

## Ghi chu van hanh

- Neu label la polygon/mask, hay chay pipeline voi `--task segment`.
- Neu label chi la bbox, dung `--task detect`.
- Public repo nay train nhanh bang `khoanhvungla/dataset.yaml`.
- `data/yolo/dataset.yaml` se duoc sinh tu dong o buoc build khi ban muon rebuild dataset theo pipeline.
- `train_yolo.py` chi la diem vao train de san. Project nay dung o buoc chuan bi.

## Ket luan

Project nay duoc thiet ke de dua du lieu ve trang thai san sang train, nhung khong tu dong train. Sau khi chay xong cac buoc tren, ban se co dataset YOLO, config train an toan, va lenh train mau de su dung thu cong khi da san sang.
