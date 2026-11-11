@echo off
set "MSYS2_PATH_TYPE=inherit"
set "CHERE_INVOKING=1"
"D:\msys2\usr\bin\bash.exe" --login -c "cd /d/Codes/ANLP/reading_project/GloVe && python3 probe_vectors.py vectors.txt %*"
