@echo off
set "MODE=%~1"
if "%MODE%"=="" set "MODE=tiny"
set "MSYS2_PATH_TYPE=inherit"
set "CHERE_INVOKING=1"
"D:\msys2\usr\bin\bash.exe" --login -c "cd /d/Codes/ANLP/reading_project && bash run_glove_local.sh %MODE%"
