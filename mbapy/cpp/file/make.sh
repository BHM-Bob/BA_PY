# chmod +x make.sh

# set wdir
WDIR=$(cd $(dirname $0); pwd)
cd "$WDIR"
echo "wdir: $WDIR"

# set storage dir
STORAGE=$(cd $(dirname $0); cd ../../../storage; pwd)
echo "storage dir: $STORAGE"

# set build dir
rm -r build
echo "build dir deleted"

# cmake, build to build dir
cmake -S . -B build
cd build
make
echo "built to build dir"

# copy built file
mv $WDIR/build/libfile.so $STORAGE/libfile.so
echo libfile.so moved to storage dir
