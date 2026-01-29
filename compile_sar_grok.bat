@echo off
setlocal

:: 设置项目目录
set PROJECT_DIR=SAR_GROK
set MAIN_FILE=main

echo [1/3] Entering directory %PROJECT_DIR%...
cd %PROJECT_DIR%

echo [2/3] Running xelatex (Pass 1)...
xelatex -interaction=nonstopmode %MAIN_FILE%.tex

:: 如果需要参考文献，可以取消下面两行的注释
:: echo Running bibtex...
:: bibtex %MAIN_FILE%

echo [3/3] Running xelatex (Pass 2)...
xelatex -interaction=nonstopmode %MAIN_FILE%.tex

echo Done! PDF should be available in %PROJECT_DIR%\%MAIN_FILE%.pdf

pause

