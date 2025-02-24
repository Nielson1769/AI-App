@echo off
REM -------------------------------------------------
REM AUTOMATE_GITHUB_DEPLOY.BAT
REM This script initializes a Git repository (if needed),
REM commits all your files, creates a GitHub repository using
REM GitHub CLI, and pushes your code.
REM -------------------------------------------------

REM Prompt for repository name.
set /p REPO_NAME="Enter your desired GitHub repository name: "

REM Check if a .git folder exists. If not, initialize a Git repo.
IF NOT EXIST ".git" (
    echo Initializing Git repository...
    git init
) ELSE (
    echo Git repository already initialized.
)

REM Add all files and commit.
git add .
git commit -m "Initial commit for LM Studio AI App"

REM Create the GitHub repository and push the code.
REM --public makes the repo public; remove it for a private repository.
gh repo create %REPO_NAME% --public --source=. --remote=origin --push

echo.
echo Repository %REPO_NAME% created and code pushed to GitHub.
echo.
echo Now, go to https://share.streamlit.io/ to deploy your app from your repository.
pause
