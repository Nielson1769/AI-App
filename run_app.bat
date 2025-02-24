@echo off
REM -------------------------------
REM Automation Script for Enhanced LM Studio AI GUI
REM -------------------------------

REM Step 1: Create a virtual environment if not present.
IF NOT EXIST "venv\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv venv
) ELSE (
    echo Virtual environment exists.
)

REM Step 2: Activate the virtual environment.
call venv\Scripts\activate

REM Step 3: Upgrade pip and install dependencies.
echo Upgrading pip and installing required packages...
python -m pip install --upgrade pip
pip install streamlit transformers torch

REM Step 4: Launch the Streamlit app.
echo Launching the AI Interface...
streamlit run app.py

pause
