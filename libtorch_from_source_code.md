下面给一套比较“稳”的做法，目标是在 **树莓派 4B（ARM64）上，从 PyTorch 源码编译出 libtorch（C++ 前端）**。我按“在树莓派上本机编译”的思路写，交叉编译就复杂很多，一般不推荐 unless 你特别熟 ARM 交叉工具链。

> 说明：不同 PyTorch 版本/系统镜像细节略有差异，下面以 **Ubuntu 20.04/22.04 + PyTorch 2.x** 为例；Raspberry Pi OS 也类似，包名可能略有不同。

---

## 一、基本思路

PyTorch 官方不提供 ARM 的 libtorch 预编译包，所以要：

1. 在树莓派上装齐依赖（C++17 编译器、CMake、Python、BLAS、CUDA/ROCm 一般木有，就走 CPU-only）。
2. 从 GitHub 拉源码。
3. 用 Python 驱动 CMake 构建：

   * 一次编译 Python 版（会顺带编 C++ core）
   * 或者直接走 `BUILD_PYTORCH_MOBILE=0` / `BUILD_LIBTORCH=1` 一类的选项。
4. 编译完成后，从 `build` 目录里抽出 C++ 静态/动态库和头文件，相当于自制 libtorch 包。

---

## 二、准备环境

### 1. 更新系统

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装基础编译工具

```bash
sudo apt install -y git build-essential cmake \
    python3 python3-pip python3-dev \
    libopenblas-dev liblapack-dev \
    libjpeg-dev zlib1g-dev \
    libpng-dev
```

* **gcc/g++ ≥ 9** 基本支持 C++17，足够编 2.x 版本的 PyTorch。
* OpenBLAS 是 CPU 线性代数库，树莓派上用它就行。

可选：如果你打算用 `ninja` 加速构建：

```bash
sudo apt install -y ninja-build
```

---

## 三、获取 PyTorch 源码

### 1. 克隆仓库

树莓派性能一般，建议固定一个 release 分支，不要整 master。

```bash
cd ~
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
# 切一个稳定 tag，举例 v2.3.0（按你需要改）
git checkout v2.3.0
git submodule sync
git submodule update --init --recursive
```

> 如果你已经 clone 过但忘记 `--recursive`，可以补：
>
> ```bash
> git submodule update --init --recursive
> ```

---

## 四、Python 依赖（构建工具）

PyTorch 的 C++ 部分实际上由 Python 的 build 脚本驱动。先装 Python 依赖：

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

如果网络原因 requirements 全装有困难，可以只装核心几个：

```bash
pip3 install typing-extensions pyyaml numpy setuptools wheel \
    cmake ninja
```

> `numpy` 用来生成一些代码/测试，不是运行时必须，但构建过程会用到。

---

## 五、配置环境变量（CPU-only）

树莓派基本上都是 CPU-only，禁用 CUDA/ROCm/MPS，否则会在配置阶段就失败。

在 `pytorch` 源码根目录下：

```bash
export USE_CUDA=0
export USE_ROCM=0
export USE_MPS=0
export USE_NCCL=0
export USE_DISTRIBUTED=0        # 如果你不需要分布式训练
export USE_QNNPACK=0            # 树莓派上常见问题，可以先关掉
export BUILD_TEST=0             # 不编译测试，减少时间
```

可选：切换使用 `ninja` 构建：

```bash
export CMAKE_GENERATOR=Ninja
```

---

## 六、编译方式选择

有两条典型路子：

1. **先编 Python 包**：`python3 setup.py develop` 或 `pip install -e .`

   * 优点：流程最成熟，文档多；同时得到 Python + C++。
   * 缺点：时间长，资源吃得多。
2. **只编 C++（libtorch）**：通过环境变量或 CMake 选项只打开 C++ 前端。

推荐的“保险路线”：**先完整编一次 PyTorch（CPU-only）**，再从 build 目录里打包 libtorch。

---

## 七、编译 PyTorch（会顺带构建 C++ core）

在 `pytorch` 根目录：

```bash
python3 setup.py develop
# 或者
# python3 setup.py install  # 如果你不需要编辑安装
```

> * 这一步在树莓派 4B 上可能会非常慢（按 SD 卡速度、散热、版本不同有数小时量级）。
> * 建议全程保持良好散热，避免降频。

如果你希望最大限度减小内存压力，可以强制单线程：

```bash
export MAX_JOBS=2  # 或 even 1
python3 setup.py develop
```

> 不要让 `MAX_JOBS` 太大，4GB 内存的 Pi 很容易 OOM。建议 1–2 之间试。

编译结束后，你应该已经能在 Python 中 `import torch`。这说明 C++ core 基本也构建完了。

---

## 八、导出/整理 C++ libtorch（自制包）

编译完成后，构建产物在 `pytorch/build` 相关目录。典型结构：

* 头文件：`pytorch/torch/include/`
* 动态库：

  * `pytorch/build/lib/libtorch.so`
  * `pytorch/build/lib/libc10.so`
  * 以及若干其他 `.so`（`libtorch_cpu.so` 等）

你可以自己组织一个“libtorch-like”目录结构，例如：

```bash
cd ~/pytorch
mkdir -p ~/libtorch_arm/ \
    ~/libtorch_arm/include \
    ~/libtorch_arm/lib

# 拷贝头文件
cp -r torch/include/* ~/libtorch_arm/include/

# 拷贝库文件（根据实际 build/lib 里名字调整）
cp build/lib/libtorch*.so ~/libtorch_arm/lib/
cp build/lib/libc10*.so ~/libtorch_arm/lib/
cp build/lib/libcaffe2*.so ~/libtorch_arm/lib/ 2>/dev/null || true
```

最终得到的 `~/libtorch_arm` 目录就可以在 C++ 项目中像官方 libtorch 那样使用。

---

## 九、在 C++ 工程中使用（CMake 示例）

假设你有一个简单 `main.cpp`：

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor t = torch::rand({2, 3});
    std::cout << t << std::endl;
    return 0;
}
```

在同一目录写 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/home/pi/libtorch_arm")  # 改成你实际路径

find_package(Torch REQUIRED)

add_executable(torch_test main.cpp)
target_link_libraries(torch_test "${TORCH_LIBRARIES}")
# 避免 RPATH 被清空
set_property(TARGET torch_test PROPERTY CXX_STANDARD 17)
```

构建：

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/home/pi/libtorch_arm ..
cmake --build .
```

运行时注意动态库搜索路径：

```bash
export LD_LIBRARY_PATH=/home/pi/libtorch_arm/lib:$LD_LIBRARY_PATH
./torch_test
```

---

## 十、如果想“只编 C++/libtorch”，不用 Python

严格意义上 PyTorch 源码目前高度依赖 Python 的 build system，纯 CMake 入口不算稳定。不过有一些可试的方式（依版本变化略大）：

1. 设置环境变量，关闭 Python binding：

   ```bash
   export BUILD_PYTHON=0
   export BUILD_SHARED_LIBS=ON
   export USE_CUDA=0
   export USE_DISTRIBUTED=0
   # 其它同前
   ```

2. 手动用 CMake：

   ```bash
   mkdir build-libtorch && cd build-libtorch

   cmake .. \
     -DBUILD_PYTHON=OFF \
     -DUSE_CUDA=OFF \
     -DUSE_DISTRIBUTED=OFF \
     -DBUILD_SHARED_LIBS=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/home/pi/libtorch_arm

   cmake --build . --target install -- -j2
   ```

   * 如果 CMake 报找不到某些 Python 组件、或脚本引用了 Python 模块，这是因为版本间依赖没完全解耦，你可能需要手动改少量 CMake 脚本或者退回“先执行 `python setup.py`”的方案。

---

## 十一、内存与性能建议（树莓派环境很重要）

1. **交换分区 / swap**
   若你的 Pi 只有 2GB/4GB RAM，编译大版本 PyTorch 可能会 OOM。可以先调大 swap：

   ```bash
   sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
   sudo /etc/init.d/dphys-swapfile stop
   sudo /etc/init.d/dphys-swapfile start
   ```

2. **最大并行度**
   `MAX_JOBS=1` 或 `2`，不要开满 4 核，否则内存压力+热问题非常明显。

3. **Release 构建**
   `CMAKE_BUILD_TYPE=Release`，Debug 会巨大且慢。

---

## 十二、可能遇到的典型问题与解决思路

* **CMake 版本太低报错**
  用 `apt` 装的 CMake 太老，可以：

  ```bash
  pip3 install cmake
  ```

  然后把 `~/.local/bin` 加到 `PATH` 里，优先用新版本。

* **链接时找不到 OpenBLAS/LAPACK**
  检查：

  ```bash
  dpkg -l | grep openblas
  dpkg -l | grep lapack
  ```

  确认已经装 `libopenblas-dev`、`liblapack-dev`，且 `cmake` 输出日志里有检测到 BLAS。

* **运行 C++ 可执行文件时找不到 libtorch.so**
  确认：

  ```bash
  export LD_LIBRARY_PATH=/home/pi/libtorch_arm/lib:$LD_LIBRARY_PATH
  ```

  或者在 CMake 里设置 RPATH。



这个报错的意思很简单：你的系统（Debian/Ubuntu 系，Python 3.12）现在启用了 **PEP 668 外部管理环境**，不允许你在系统级 Python 里直接用 `pip3 install` 往 `/usr/lib/python3.12` 里面乱装包，怕把系统搞炸。

**解决思路：给 PyTorch 建一个自己的虚拟环境，在虚拟环境里用 pip 装依赖和编译。**
下面直接给你一套完整的命令，从你现在的 `pytorch` 目录开始用。

---

## 1. 安装 venv 支持（只要做一次）

先确认系统有虚拟环境模块：

```bash
sudo apt update
sudo apt install -y python3-venv python3-full
```

如果已经安装过，会提示已是最新版本，可以忽略。

---

## 2. 在 PyTorch 源码目录创建虚拟环境

你当前在：

```bash
/media/linxi-ice-2/CCCOMA_X64FRE_ZH-CN_DV91/test1/libtorch_creater/pytorch
```

就在这个目录下建一个 `.venv`：

```bash
cd /media/linxi-ice-2/CCCOMA_X64FRE_ZH-CN_DV91/test1/libtorch_creater/pytorch

# 创建虚拟环境（目录名可以自定义，这里用 .venv）
python3 -m venv .venv
```

---

## 3. 激活虚拟环境并安装依赖

每次要编译 / 开发 PyTorch，都要先激活这个环境：

```bash
cd /media/linxi-ice-2/CCCOMA_X64FRE_ZH-CN_DV91/test1/libtorch_creater/pytorch

# 激活虚拟环境
source .venv/bin/activate
```

成功的话，命令提示符前面会多一个 `(.venv)`。

然后在**虚拟环境内**执行 pip：

```bash
pip install --upgrade pip

# 如果网络不好 requirements.txt 全量装不动，可以先试这一句
pip install -r requirements.txt
```

如果 `requirements.txt` 太多、某些包装不下来，也可以退而求其次，只装基本构建依赖：

```bash
pip install typing-extensions pyyaml numpy setuptools wheel \
    cmake ninja
```

> 注意：只要你看到提示里写的是类似
> `(.venv) linxi-ice-2@...`
> 再用 pip 就不会再有 `externally-managed-environment` 报错了，因为这时 pip 只在 `.venv` 目录里工作，不会碰系统 Python。

---

## 4. 在虚拟环境内编译 PyTorch

仍然保持虚拟环境已激活（前面有 `(.venv)`），设置好环境变量，走 CPU-only 编译：

```bash
cd /media/linxi-ice-2/CCCOMA_X64FRE_ZH-CN_DV91/test1/libtorch_creater/pytorch

# 关闭 GPU / 分布式等
export USE_CUDA=0
export USE_ROCM=0
export USE_MPS=0
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_QNNPACK=0
export BUILD_TEST=0

# 降低并行度防止树莓派内存爆掉
export MAX_JOBS=2

# 可选：用 Ninja 提速（前提是 apt 安装了 ninja-build）
export CMAKE_GENERATOR=Ninja

# 开始编译（develop 或 install 选一个）
python setup.py develop
# 或：
# python setup.py install
```

**关键点：** 这里的 `python` 一定要是虚拟环境里的 `python`（激活之后默认就是 `.venv/bin/python`），这样它会自动使用你刚才通过 `pip` 装的那些构建依赖。

---

## 5. 导出 / 使用 libtorch

编译成功后，你可以像之前说的那样，从 `pytorch/torch/include` 和 `pytorch/build/lib` 里把头文件和 `.so` 文件拷出来，组织成一个 `libtorch_arm` 目录，再在 CMake 里通过 `CMAKE_PREFIX_PATH` 使用。

---

## 6. 如果你以后重新开机 / 新终端

每次要继续编译或使用这个 Python 环境，只需要在 `pytorch` 目录下重新激活：

```bash
cd /media/linxi-ice-2/CCCOMA_X64FRE_ZH-CN_DV91/test1/libtorch_creater/pytorch
source .venv/bin/activate
```

然后再运行 `python setup.py ...` 或其他脚本即可。

---




下面给出 **最标准、最干净的 pip config 换源方式**（适用于你的树莓派 + venv 环境）。
全部使用 **pip config set** 命令，不需要手动编辑文件。

---

# ✅ 一、虚拟环境中使用 `pip config` 换源（你当前编译 PyTorch 必须用这个）

首先进入你的 `.venv`：

```bash
cd /media/linxi-ice-2/CCCOMA_X64FRE_ZH-CN_DV91/test1/libtorch_creater/pytorch
source .venv/bin/activate
```

然后执行 pip config：

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
```

完成后测试：

```bash
pip config list
```

应该看到类似：

```
global.index-url='https://mirrors.aliyun.com/pypi/simple/'
global.trusted-host='mirrors.aliyun.com'
```

此时 pip install 已默认走阿里云镜像。

---

# ✅ 二、如果你想恢复官方源（可选）

```bash
pip config unset global.index-url
pip config unset global.trusted-host
```

---

# ✅ 三、pip config 会写到哪里？

在虚拟环境下，pip config 写入此处：

```
<你的 venv 目录>/pip.conf
```

不会污染系统 Python，也不会触发 “externally-managed-environment”。

---

# ✅ 四、测试是否成功

```bash
pip install numpy
```

看到：

```
Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
```

即成功换源。






