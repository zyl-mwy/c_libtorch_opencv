# Usage of c++ libtorch in Ubuntu System

From [知乎](https://zhuanlan.zhihu.com/p/513571175) and [csdn](https://blog.csdn.net/liang_baikai/article/details/127849577)

## Prepare
1. Download LibTorch and unzip it
```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip

unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip
```
2. Install opencv
```bash
sudo apt install libopencv-dev
```
3. Move the libtorch directory to project directory
```bash
mv
```
4. Run python program about deeplearning and coverting
```
python digit.py

python test_digit.py
```
5. Run c++ program by cmake and make
```
rm -rf build

mkdir build

cmake ..

make

./digit ../model/digit.jit ../image/sample.png
```
