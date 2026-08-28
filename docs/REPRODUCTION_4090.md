# 单张 RTX 4090 复现入口

本目录为当前 fork 增加可移植的低显存运行方式，不替换上游方法。原始运行已经在一张 RTX 4090 上完成；**本次整理后的安装器、启动器尚未重新完成端到端安装和 GPU 推理验证**，目前只做了源码、语法及 CPU 检查。功能范围和待验证边界见 [功能与边界](CAPABILITIES_AND_LIMITS.md)。

## 已完成的基线

以下数字取自迁移前成功运行的统计 JSON，不是新脚本的测速结果，也不是硬件最低要求或论文指标：

| 项目 | 原运行记录 |
| --- | --- |
| 模型 | Wan2.1-T2V-1.3B-Diffusers |
| 输出 | 81 帧，480 高 × 832 宽，20 FPS |
| 采样 | 50 步，seed=25234，mask-step=30，fix-rope-step=5 |
| GPU / 精度 | 单张 RTX 4090；主模型 BF16，VAE FP32 |
| 内存策略 | CPU model offload；无 VAE 空间 tiling |
| 软件 | Torch 2.7.1+cu118，Diffusers 0.33.1 + 作者 transformer |
| 模型加载 / 推理 | 28.126 秒 / 552.465 秒 |
| GPU 峰值 allocated / reserved | 13.657 GiB / 13.863 GiB |
| 进程峰值 RSS | 19.693 GiB |

推理时间不含下载、环境安装，也不含之后的视频导出。主存 RSS 不是整机内存需求；GPU 统计不是最小显存证明。原始 JSON 中的服务器路径和 GPU UUID 不随公开仓库发布。视频可生成、可解码，不等于已经复现论文的轨迹命中率或质量指标。

## 前提与隔离

- Linux x86_64、glibc ≥ 2.28、Python 3.11、可用的 NVIDIA 驱动、Bash 和 `flock`。
- 一张经管理员或调度器允许使用的 GPU。默认空闲检查要求至少 23,000 MiB 可用显存；这是一项保守启动门槛，不是显存上限或最小需求。
- CPU offload 需要足够空闲主存；建议预留至少 32 GiB 主存和约 50 GiB 磁盘作为起点，再按本机环境留余量。仅八个模型权重分片合计约 26.92 GiB。
- 安装器只操作指定工作目录中的独立 `.venv`，使用 [requirements-4090.txt](../requirements-4090.txt) 的直接依赖版本；这不是完整的传递依赖锁文件。不会安装系统 CUDA、升级驱动或使用全局 pip。

Python 3.11 不存在时，先用你自己的环境管理工具安装用户级 Python，再向安装器传入其路径。不要为本项目修改共享服务器的系统 Python 或驱动。

## 安装当前 checkout

下面的命令均在本仓库根目录运行。工作目录默认是当前 checkout 的 `.repro`，也可以换成自己有权限的绝对路径。

```bash
WORKDIR="$PWD/.repro"
bash repro/setup_4090.sh --workdir "$WORKDIR" --python python3.11
RUNTIME="$WORKDIR/.venv/bin/python"
MODEL="$WORKDIR/models/Wan2.1-T2V-1.3B-Diffusers"
```

安装器不会再次 clone 上游仓库。它将当前 checkout 的 `module/transformer_wan.py` 写入这个私有 venv 的 Diffusers 0.33.1，首次保存原文件为 `.ditraj-original`，随后比较 SHA-256。不是通过修改全局 Diffusers 来使项目运行。

如果以后修改了 checkout 中的 transformer，需要明确同步补丁，再运行：

```bash
"$RUNTIME" -I repro/apply_wan_patch.py --venv "$WORKDIR/.venv"
"$RUNTIME" -I repro/apply_wan_patch.py --venv "$WORKDIR/.venv" --check
```

已有独立 venv 可以通过启动器的 `--python` 复用。若它使用 venv 外的 editable Diffusers，`--check` 会只读定位实际源码并比较哈希；写入模式仍拒绝修改外部源码。它必须已经是 0.33.1 且与本 checkout 补丁一致，否则请用新的工作目录安装，或自行管理那个私有源码树。不会自动改动现有运行环境。

## 下载或复用模型

模型固定到官方提交 [0fad780a534b6463e45facd96134c9f345acfa5b](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/0fad780a534b6463e45facd96134c9f345acfa5b)。安装器不自动下载权重；在有网络的机器上单独运行：

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
  "$WORKDIR/.venv/bin/hf" download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --revision 0fad780a534b6463e45facd96134c9f345acfa5b \
  --local-dir "$MODEL"
```

下载中断后可重跑同一命令；必须检查命令成功退出，不能把进度条百分比当成全部分片已完成的证据。该命令固定来源版本，但不是独立的逐文件 SHA-256 审计工具。已有完整权重目录可以直接传给 `--model`，不需要复制进仓库。请保留模型原有目录结构及配置文件。

启动器始终使用本地模型并启用 Hugging Face 离线模式；缺文件时会报错，不会在 GPU 推理中悄悄补下载。

## 不占 GPU 的检查

这一步只读取参数、prompt 和代码指纹，不查询 GPU、不加载权重、不创建输出目录：

```bash
bash repro/run_4090.sh --workdir "$WORKDIR" --model "$MODEL" --dry-run
```

不安装模型环境也可以用已有 Python 做同类检查：

```bash
python3 -B repro/run_low_vram.py --dry-run --output /tmp/ditraj-dry-run.mp4
```

`--dry-run` 不证明依赖、模型、CUDA、显存或底层小框约束可用；其 JSON 是拟运行配置，不是测量结果。

## 启动推理

先取得相应 GPU 的使用权限。默认 GPU 是物理编号 0；例如明确选用编号 5：

```bash
bash repro/run_4090.sh --workdir "$WORKDIR" --model "$MODEL" --gpu 5 --mode smoke
bash repro/run_4090.sh --workdir "$WORKDIR" --model "$MODEL" --gpu 5 --mode full
```

两个命令分别创建新的运行目录；只需按需执行其中之一。`smoke` 是 9 帧、160×288、2 步、控制截止步 1/1 的排错配置，不是质量评测。`full` 使用上面的基线默认参数。

启动器会检查计算进程、显存使用和 GPU 利用率，取得当前工作目录内的每 GPU `flock` 后，将所选物理卡的 UUID 暴露为进程内唯一的 CUDA GPU。检测到占用即拒绝启动，不会杀进程。这个检查与锁**不能替代集群预约**，也不能阻止其他用户或其他工作目录的进程随后使用同一张卡。

其他 runner 参数放在 `--` 后；模型、输出、Python 和 GPU 使用启动器参数：

```bash
bash repro/run_4090.sh \
  --workdir "$WORKDIR" --model "$MODEL" --python "$RUNTIME" --gpu 0 \
  --output "$WORKDIR/outputs/my-trial/output.mp4" \
  -- --seed 123 --steps 10 --prompt-index 0
```

使用新的输出路径；主视频、框线视频、统计 JSON 或同目录 `run.log` 已存在时会拒绝覆盖。输出包含 `output.mp4`、`output_box.mp4`、`output.metrics.json` 和 `run.log`。后两者可能包含本机路径、GPU 信息与错误细节，不要直接加入公开提交。

默认轨迹仍是等尺寸框从左向右移动；此入口没有自定义轨迹 CLI，也没有新增多物体、图片输入或 3D/4D 功能。改变尺寸、时长、控制截止步等不代表已验证这些组合的效果或资源上限。

## CPU 回归与检查记录

以下测试不需要 Torch、模型下载或 GPU：

```bash
python3 -B -m unittest discover -s tests -v
bash -n repro/setup_4090.sh
bash -n repro/run_4090.sh
python3 -B repro/apply_wan_patch.py --help
python3 -B repro/gpu_preflight.py --help
```

当前 CPU 套件共 25 项：20 项通过，5 项为明确标记的上游已知边界缺陷（`expected failures`），不是把错误修好了。这些测试主要覆盖参数和轨迹纯函数，不能替代安装器、GPU 占用检查、文件锁、补丁写入及真实模型推理的集成测试。新打包脚本尚未完成清洁环境端到端复测。

`.repro/`、虚拟环境、模型、缓存、日志及生成视频不应入库；代码变更前后应保留自己的运行参数与资源统计，再按 [功能与边界](CAPABILITIES_AND_LIMITS.md) 中的未执行测试矩阵逐项验证。
