# GitHub Upload Commands

Run the following commands from the repository root:

```bash
cd /home/xinxueshi/yxy/program/RAXO/RMD_X_GitHub_upload

# Optional safety check.
find . -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.pkl' -o -name '*.parquet' -o -name '*.zip' -o -name '*.tar' -o -name '*.7z' -o -name '*.rar' \)
find . -type f -size +50M

# Initialize Git if this folder is not already a Git repository.
git init
git branch -M main

# Add the target remote. If the remote already exists, use set-url instead.
git remote add origin https://github.com/YXY-PRIV/RMD-X.git
# git remote set-url origin https://github.com/YXY-PRIV/RMD-X.git

# Commit and push.
git add .
git commit -m "Release RMD-X code and reproduction documentation"
git push -u origin main
```

If Git asks for authentication, use a GitHub personal access token instead of
your account password.

