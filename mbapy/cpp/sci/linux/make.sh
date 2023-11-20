cd /media/bhm-bob/BHM/My_Progs/BA/Python/BA_PY/mbapy/cpp/sci/linux/
rm -r build
echo remove build
cmake -S . -B build
cd build
make
echo built to build dir
mv /media/bhm-bob/BHM/My_Progs/BA/Python/BA_PY/mbapy/cpp/sci/linux/build/libsci.so /media/bhm-bob/BHM/My_Progs/BA/Python/BA_PY/mbapy/storage/libsci.so
echo moved libsci.so to storage dir