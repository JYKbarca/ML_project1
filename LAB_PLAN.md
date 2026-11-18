# DDA5001 Part I 机房实验清单

按[课程 README](p1/README.md)执行。离开机房前带走六组实验的原始记录、单组图、汇总图和至少一份文本样本，检查点另行保存。报告可以在拿到结果后撰写。

## 1. 获取代码并激活课程环境

- [ ] 在本地提交并推送实验代码、配置和本清单。第一次到机房获取仓库时，在终端运行：

```bash
git clone -b main https://github.com/JYKbarca/ML_project1.git
cd ML_project1/p1
source /opt/dda5001/venv/bin/activate
ls config
```

如果机房已经克隆仓库，在仓库根目录运行 `git pull --ff-only origin main`，再进入 `p1` 并激活课程环境。确认 `config` 下有原始配置和五份新增配置。下面的实验命令均在 `p1` 中执行。

## 2. 验证代码和训练流程

- [ ] 先运行代码检查，通过后再运行 CPU smoke test。如果 `checks/cpu` 已存在，为 smoke test 换一个新目录。

```bash
python check_solution.py && python train.py config/smoke.py --device=cpu --out_dir=checks/cpu
```

- [ ] 两项验证均成功后再开始正式训练。

## 3. 完成六次正式训练

- [ ] 用 `nvidia-smi` 确认 GPU 和显存，用 `df -h .` 确认有空间保存检查点。默认配置约需 20 GiB 显存。如果需要适配，按[环境说明](p1/setup/README.md)在复制的配置中统一调整六组硬件参数，并记录实际变更供报告使用。若为 AdamW 默认组另存了适配配置，第一条命令改用该配置，输出仍为 `runs/adamw-default`。
- [ ] 在同一张 GPU 上按顺序执行，每次运行一条，上一组完成后再启动下一组。先用 AdamW 默认组的实测耗时估算剩余工作。

```bash
python train.py config/train_shakespeare_char.py  # AdamW 默认 0.0003
python train.py config/adamw-other.py             # AdamW 自选 0.0006
python train.py config/muon-default.py            # Muon 默认 0.01
python train.py config/muon-other.py              # Muon 自选 0.02
python train.py config/sgdm-default.py            # SGDM 默认 0.1
python train.py config/sgdm-other.py              # SGDM 自选 0.2
```

- [ ] 每组完成后检查 `summary.json` 中的 `status` 为 `completed`，`step` 为 `5000`，`tokens` 为 `92160000`。确认 `metrics.csv`、`config.json`、`environment.json`、`run.log` 和 `last.pt` 均已保存。

## 4. 绘制单组图并生成文本

- [ ] 六组分别绘图，每组有训练和验证损失随 token 数变化、随训练时间变化的两张图。以下循环在六组齐全后执行。如果有错误，解决后再继续。

```bash
for run in adamw-default adamw-other muon-default muon-other sgdm-default sgdm-other; do
  python plot.py --run-dir="runs/$run" || break
done
```

- [ ] 生成至少一份代表性文本。课程要求一份，本例使用 AdamW 默认组的 `last.pt`，文本会保存到同组的 `samples.txt`。

```bash
python sample.py --checkpoint=runs/adamw-default/last.pt --tokens=400
```

如果希望对照六组的生成效果，可以用下面的循环代替上一条命令。报告中选一份并注明来源。

```bash
for run in adamw-default adamw-other muon-default muon-other sgdm-default sgdm-other; do
  python sample.py --checkpoint="runs/$run/last.pt" --tokens=400 || break
done
```

## 5. 生成汇总图、导出并备份

- [ ] 将失败或额外实验移到 `runs/` 外保存，确保汇总只读取本次六组正式结果。
- [ ] 绘制汇总图并导出 ZIP。已有同名 ZIP 时，使用新文件名。

```bash
python plot.py --runs-dir=runs && python export_results.py --runs-dir=runs --output=results01.zip
```

### 一次执行第 4–5 节的命令

六组训练都完成后，在已激活课程环境的 Ubuntu 终端中进入 `p1`，将下面的命令整段复制执行。它会依次绘制六组单独图、用 AdamW 默认组生成一份文本、绘制两张汇总图并导出 ZIP。任一步报错，后续命令会停止。如果 `results01.zip` 已存在，先把最后一条命令的输出文件名改为新文件名。

```bash
(
  set -e

  for run in adamw-default adamw-other muon-default muon-other sgdm-default sgdm-other; do
    python plot.py --run-dir="runs/$run"
  done

  python sample.py --checkpoint=runs/adamw-default/last.pt --tokens=400
  python plot.py --runs-dir=runs
  python export_results.py --runs-dir=runs --output=results01.zip
)
```

分步执行或整段执行任选一种。导出成功后按下面的清单验收和备份。

- [ ] 打开两张汇总图，确认每张都包含六组验证损失曲线。横轴分别是训练 token 数和训练时间。
- [ ] 打开 ZIP，确认包含代码、配置、六组实验记录、12 张单组图、2 张汇总图和至少一份 `samples.txt`。把 ZIP 复制到自己的存储，模型检查点另行备份。
- [ ] 如果下午跑不完，备份已完成的实验目录和检查点，记下剩余组别。下次只跑未完成的组；汇总时把六组齐全的正式结果放回同一个 `runs/` 目录。

## 6. 在哪里找到结果，怎样带回家

以下是训练、绘图、文本生成和导出都完成后的目录结构。其他五组与 `adamw-default` 的结构相同。若只给 AdamW 默认组生成文本，只有该组有 `samples.txt`。

```text
ML_project1/
├── LAB_PLAN.md
└── p1/
    ├── results01.zip
    └── runs/
        ├── adamw-default/
        │   ├── config.json
        │   ├── environment.json
        │   ├── metrics.csv
        │   ├── summary.json
        │   ├── run.log
        │   ├── best.pt
        │   ├── last.pt
        │   ├── loss_vs_tokens.png
        │   ├── loss_vs_train_seconds.png
        │   └── samples.txt
        ├── adamw-other/
        ├── muon-default/
        ├── muon-other/
        ├── sgdm-default/
        ├── sgdm-other/
        ├── comparison_tokens_vs_tokens.png
        ├── comparison_tokens_vs_train_seconds.png
        └── comparison.json
```

### 方式一：直接带走结果包

- [ ] 在机房把 `p1/results01.zip` 复制到 U 盘或上传到自己的网盘。回家解压后，在 ZIP 内的 `results/` 查看六组实验的指标、图和样本。写报告所需的材料都在里面。
- [ ] 如果之后还需要加载模型或重新采样，将相应的 `best.pt` 或 `last.pt` 另行备份。导出器会排除模型检查点。

### 方式二：通过 GitHub 同步写报告的材料（可选）

仓库默认忽略 `runs/`、`*.zip`、`*.pt` 和 `*.log`。要通过 GitHub 获取 CSV、JSON、图和样本，在**机房**先将 ZIP 内的 `results/` 解压到仓库根目录，使它与 `LAB_PLAN.md` 同级。只解压 `results/` 即可。确认机房的 GitHub 认证有仓库写权限，然后在仓库根目录执行：

```bash
git add -- results/
git diff --cached --stat
git commit -m "Add Part I experiment results"
git push origin main
```

回家后在本地仓库根目录运行 `git pull --ff-only origin main`，并在 `results/` 查看结果。完整的提交包 `p1/results01.zip` 仍按方式一带走，方便后续加入报告。

## 7. 撰写报告并提交

- [ ] 根据保存的配置和环境记录说明模型、训练预算、学习率、实际硬件及适配设置。
- [ ] 比较每种优化器使用两个学习率时的收敛速度、波动和最终验证损失。结合汇总图，按相同 token 数和相同训练时间比较各组，说明观测范围内是否出现曲线交叉。
- [ ] 引用代表性文本片段并注明来源。将报告 PDF 加入结果 ZIP，按 handbook 和 Blackboard 核对提交要求。
