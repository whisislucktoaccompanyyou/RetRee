# ReTree：用扩散模型做视网膜图像生成与分割

本仓库基于论文 *Denoising Diffusion Probabilistic Model for Retinal Image Generation and Segmentation*（ICCP 2023）的实现。本地改动的目标是：用血管掩膜生成眼底图，也就是把伪标签转回医学图像。作者原有的训练脚本仍保留。

![方法示意图](ddpms.jpg)

- 数据集：[Kaggle ReTree dataset](https://www.kaggle.com/datasets/alnuritoalimanov/retree-dataset)
- 作者权重：[Google Drive](https://drive.google.com/drive/folders/1WSPo2yUUmy9o4rCB7h7lQkazcl77GHMf?usp=drive_link)

## 作者原始流程

1. 用 `conda env create -f env.yml` 创建环境。
2. 下载 EyeQ、DRIVE、STARE、CHASE DB1，放到 `datasets/`。
3. 用任意分割模型给 EyeQ 做分割。
4. 把 EyeQ 图像放到 `datasets/eyeq/images/`。
5. 运行 `train_ddpm_mask.py`，把生成的掩膜放到 `datasets/eyeq/labels/`。
6. 运行 `train_ddpm_image.py`。

作者脚本把路径写死在 EyeQ 上，默认走推理，并且按文件名排序配对图像和掩膜。本地没有作者权重，也不能直接接到现有的 DRIVE 和 Kaggle 目录上，所以下面的入口是另写的，没有改作者脚本里的损失和网络结构。

## 本地改了什么，以及为什么

| 文件 | 做了什么 | 为什么 |
| --- | --- | --- |
| `run_retree.py` | 按 DRIVE 编号配对图像和血管图，训练掩膜条件图像 DDPM（分割图到眼底图） | 作者脚本按 glob 顺序配对，编号对不上就会训错。这一步才是“血管图转回眼底图” |
| `train_pipeline.py` | 先在 DRIVE 上训一个小分割器，再给 Kaggle 眼底图打伪标签，最后用 DRIVE 真配对加伪配对训练 DDPM | Kaggle 只有眼底图、没有血管图，不能单独监督掩膜到图像的模型。DRIVE 只有 20 张有标注的训练图，测试集没有血管标注，所以留出分数来自这 20 张的末尾 4 张 |
| `infer_retree.py` | 读 `<id>_mask.png` 和 `<id>_image.png`，输出原图、掩膜、生成图的对比条，以及单独的生成图 | 用来检查伪标签能不能转回眼底图。原图只用于对比，不参与生成 |
| `train_background.sh` | 用 `setsid` 后台启动 `train_pipeline.py`，关掉 SSH 不会停。已有进程时拒绝再起一份 | 训练时间长，不能绑在一个终端上 |
| `test_input/README.txt` | 说明放入一对 `image.png` 和 `mask.png` | 方便单独做一次生成检查 |

具体取舍：

- 权重只存模型参数，并且每个阶段只留一个滚动文件。早期每个 epoch 都存优化器状态，单文件大约 1.6 GB，长跑会把磁盘写满。
- 正式采样必须走完整的 1000 步。这个反向过程假定时间步每次减 1，抽稀训练好的时间表会让更新不一致，采出来是纯噪声。`infer_retree.py` 因此要求 `--sample-steps` 等于 `--noise-steps`。
- `run_retree.py` 里每个 epoch 的预览图会抽稀采样。那只是为了出一张检查图，不代表正式生成质量。
- 噪声日程、MSE 加 L1 的噪声预测损失，以及 4 通道条件输入，沿用作者的 `train_ddpm_image.py`。分割器是另加的小 U-Net：DRIVE 只有 20 张标注图，大模型容易过拟合。正类权重设为 8，因为血管像素很少。
- Kaggle 图像按 EyeQ 质量表过滤后再打伪标签。这些掩膜是本地分割器的预测，不是人工标注，也不能当成作者原始标签。

## 本地训练与推理

后台训练，默认使用 0 号 GPU。启动前先确认这张卡是空的：

```bash
sh train_background.sh 0
tail -f train.log
```

只在 DRIVE 上训练掩膜到图像的模型：

```bash
python -u run_retree.py --epochs 2 --batch-size 2
```

用训好的权重复原：

```bash
python infer_retree.py \
  --input test_input \
  --output test_output \
  --checkpoint results/drive_seg2img/checkpoints/ddpm_latest.pth.tar
```

`test_input` 里放一对眼底图和血管伪标签。掩膜为单通道，背景是 0，血管是 1 或 255。

数据集、训练权重、`results/`、日志和测试图都在 `.gitignore` 里，不会提交。方法示意图 `ddpms.jpg` 会保留。

## 引用

使用本代码请引用原文：

```bibtex
@inproceedings{alimanov2023denoising,
  title={Denoising Diffusion Probabilistic Model for Retinal Image Generation and Segmentation},
  author={Alimanov, Alnur and Islam, Md Baharul},
  booktitle={2023 IEEE International Conference on Computational Photography (ICCP)},
  pages={1--12},
  year={2023},
  organization={IEEE}
}
```
