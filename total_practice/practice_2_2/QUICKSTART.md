# Activate venv & install dependencies

```powershell
# 1. Activate (PowerShell)
cd D:\Hoc_tap\TOTAL_LABPARACTICE_FOR_DEEP_LEARNING\total_practice\practice_2_2
.venv\Scripts\Activate.ps1

# Nếu gặp lỗi "running scripts is disabled on this system":
#   Mở PowerShell as Administrator, chạy:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   rồi activate lại

# 2. Cập nhật pip + cài dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Crawl ảnh
python -m craw.crawl_tiki

# 4. Làm sạch & resize
python -m data_processing.pipeline
```

## CMD (alternative)

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Bash (Git Bash / WSL)

```bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Deactivate khi xong

```bash
deactivate
```
