@echo off
echo [1/4] Creating Python virtual environment...
python -m venv venv

echo [2/4] Activating virtual environment...
call venv\Scripts\activate

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo [4/4] Installing required packages...
pip install -r requirements.txt

echo --------------------------------------
echo ✅ Setup complete!
echo To activate later: call venv\Scripts\activate
pause