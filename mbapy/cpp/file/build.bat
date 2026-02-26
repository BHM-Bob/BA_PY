@echo off

:: Set wdir
pushd "%~dp0"
set "WDIR=%CD%"
echo wdir: %WDIR%

:: Set storage dir
cd /d "%~dp0..\..\storage"
set "STORAGE=%CD%"
echo storage dir: %STORAGE%
popd

:: Set build dir
if exist build rd /s /q build
echo build dir deleted

:: cmake, build to build dir
cmake -G "MinGW Makefiles" -S . -B build
cd build
make
echo built to build dir

:: Copy built file
move "%WDIR%\build\libfile.dll" "%STORAGE%\libfile.dll"
echo libfile.dll moved to %STORAGE%
cd ..