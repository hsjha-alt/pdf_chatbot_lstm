@echo off
echo Initializing Git repository...
git init

echo Adding all files...
git add .

echo Committing files...
git commit -m "first commit"

echo Setting branch to main...
git branch -M main

echo Adding remote origin...
git remote add origin git@github.com:hsjha-alt/pdf_chatbot_lstm.git

echo Pushing to GitHub...
git push -u origin main

echo Done!
pause

