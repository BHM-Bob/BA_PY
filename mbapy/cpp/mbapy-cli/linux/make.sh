###
 # @Date: 2023-11-20 16:03:31
 # @LastEditors: BHM-Bob 2262029386@qq.com
 # @LastEditTime: 2024-01-05 17:09:54
 # @Description: 
### 

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
mv $WDIR/build/mbapy-cli $STORAGE/mbapy-cli
echo mbapy-cli moved to storage dir