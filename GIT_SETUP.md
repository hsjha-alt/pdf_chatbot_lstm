# Git Setup Instructions

## Quick Setup (Using Batch Script)

Simply double-click `setup_git.bat` in the `pdf_chatbot_lstm` folder.

## Manual Setup (Run in Terminal)

Open PowerShell or Command Prompt in the `pdf_chatbot_lstm` folder and run:

```bash
# Initialize Git repository
git init

# Add all files (respects .gitignore)
git add .

# Commit files
git commit -m "first commit"

# Set branch to main
git branch -M main

# Add remote origin
git remote add origin git@github.com:hsjha-alt/pdf_chatbot_lstm.git

# Push to GitHub
git push -u origin main
```

## Note

The `.gitignore` file has been created to exclude:
- `data/` folder (large binary files)
- `models/` folder (trained models)
- `__pycache__/` (Python cache)
- Other temporary files

If you want to include PDFs, they will be committed. If you want to exclude them, uncomment the `# pdfs/` line in `.gitignore`.

## Files Being Committed

- All Python source files (`.py`)
- `README.md`
- `requirements_new.txt`
- `run_new.bat`
- `.gitignore`
- `pdfs/` folder (PDF files)
- `setup_git.bat`

## Files Excluded (via .gitignore)

- `data/` folder (chunks, embeddings)
- `models/` folder (trained models)
- `__pycache__/` (Python cache)
- Virtual environment folders

