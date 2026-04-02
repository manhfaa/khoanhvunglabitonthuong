# Pipeline chuan bi train YOLO cho khoanh vung la bi ton thuong

Pipeline nay duoc thiet ke de chuan bi du lieu va cau hinh train YOLO an toan cho may co RTX 5060 8GB VRAM va RAM 16GB. Muc tieu la dung truoc buoc train, khong tu dong train, khong goi model lon, va khong de cau hinh mac dinh gay OOM.

## 1. Audit du lieu

- Quet `data/raw`
- Kiem tra tong so anh, anh loi, kich thuoc, do mo, do sang, phan bo nguon
- Neu co label thi thong ke phan bo class
- Xuat `outputs/audit_report.json` va `outputs/audit_report.csv`

## 2. Preprocess

- Doc anh an toan
- Sua EXIF orientation
- Chuan hoa extension
- Loai anh loi, qua nho, qua mo, qua toi, qua chay
- Resize gioi han canh dai ma van giu ti le
- Luu vao `data/cleaned`

## 3. Remove duplicates

- Phat hien duplicate chinh xac bang SHA256
- Phat hien near-duplicate bang perceptual hash
- Co 2 che do `report-only` va `apply`
- Khi `apply`, giu anh tot hon va dua anh bi loai vao `quarantine/`

## 4. Synthesize minority classes

- Sinh them mau nhe cho class it du lieu
- Khong dung API ngoai
- Neu la segmentation thi co the tach foreground theo polygon va ghep nen nhe
- Neu la detection thi uu tien synthesis nhe tu bbox/crop hien co
- Ghi metadata de danh dau `is_synthetic`, `parent_image`, `method`

## 5. Augmentation config

- Dung Albumentations
- Chi ap dung cho train
- Co cau hinh bat/tat va xac suat tung phep bien doi trong `configs/thresholds.yaml`
- Tranh augmentation qua manh lam sai hinh thai benh

## 6. Validate annotations

- Kiem tra format YOLO detect hoac segment
- Bao loi class id, bbox/polygon vuot khung, polygon khong hop le
- Xuat report chi tiet truoc khi build dataset

## 7. Build YOLO dataset

- Gop anh sach, ket qua dedup, va anh synth
- Split `train/val/test`
- Co co gang stratified theo class neu du du lieu
- Dam bao near-duplicate khong nam o nhieu split khac nhau
- Tao `data/yolo/dataset.yaml`

## 8. Generate train config

- Chon model nano phu hop GPU 8GB
- Mac dinh `imgsz=640`, `batch=8`, `workers=2`, `amp=true`, `cache=false`
- Neu du doan de OOM thi giam `batch`, sau do moi giam `imgsz`
- Tao `configs/yolo_train.yaml`

## 9. Dung o day, chua train

- Script `src/train_yolo.py` da san sang
- README co lenh train mau
- Khong tu dong goi `train()`
- Ket thuc o muc chuan bi train
