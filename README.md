# Genshin AutoFish Mini

本库基于[genshin_auto_fish](https://github.com/7eu7d7/genshin_auto_fish)精简而来，去除了自动找鱼抛竿等功能（不太好使而且慢），仅保留自动提竿、自动力度控制，毕竟后者才是钓鱼最痛苦的部分XD。

同时，移除了模型训练代码，使用Onnxruntime替代Torch，极大的精简了环境大小。

# 安装使用流程

安装python运行环境（解释器），推荐使用 [anaconda](https://www.anaconda.com/products/individual#Downloads).

## python环境配置

打开anaconda prompt(命令行界面)，创建新python环境并激活:
```shell
conda create -n ysfish python=3.8
conda activate ysfish
```

推荐安装**python3.8或以下**版本。

## 下载工程代码

使用git下载，[git安装教程](https://www.cnblogs.com/xiaoliu66/p/9404963.html):

```shell
git clone https://github.com/7eu7d7/genshin_auto_fish.git
```

或直接在**github网页端**下载后直接解压。

## 依赖库安装

切换命令行到本工程所在目录:

```shell
cd genshin_auto_fish
```

执行以下命令安装依赖:

```shell
python -m pip install -U pip
pip install -r requirements.py
```

因只保留强化训练模型，计算量极小，无需GPU加速，仅安装CPU版本的Onnxruntime即可

# 运行钓鱼AI

原神需要以1920x1080的分辨率运行，分辨率高的屏幕可以开窗口模式。

## 手动运行

打开命令行或Powershell（一定要以**管理员权限**启动），执行以下代码：

```shell
python fishing.py
```

运行后出现`Press "R" to start fishing`后按R键开始钓鱼。

与原版不同的是，每次钓上鱼后都需要按R重新开始，即“抛竿 -> 按R -> 等待程序操作”如此循环操作。

## 一键脚本

运行`start.bat`即可（可能需要对conda进行配置，按照屏幕提示操作即可）。
