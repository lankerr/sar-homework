# SAR 大作业 - 分数阶傅里叶变换应用

> **课程大作业存档**
>
> 本项目为课程作业，实验结果未达预期效果，项目已终止，不计划继续研究。

## 项目简介

本作业探索分数阶傅里叶变换（Fractional Fourier Transform, FrFT）在合成孔径雷达（SAR）成像中的应用。主要研究内容包括：

- Hermite 基底在 FrFT 中的实现
- SAR 成像算法仿真（BPA, CSA, RDA, WKA）
- 抗干扰性能分析

## 项目结构

```
├── FrFT_XZSC_GWDFF.py           # 分数阶傅里叶变换实现
├── SAR_hermite_PSL滤波器生成.py # Hermite 滤波器
├── 成像BPA,CSARDA.py            # BPA/CSA/RDA 成像算法
├── 成像仿真wka修复.py            # WKA 成像算法
├── 二维hermite仿真.py            # 二维 Hermite 仿真
├── 基于FrFT的hermite基底.py      # FrFT Hermite 基底
├── SAR_GROK/                    # 相关实验代码
├── djx-hw/                      # 作业相关文件
├── 邓嘉轩_SAR大作业.pdf           # 作业报告
└── 运行说明.txt                  # 运行说明
```

## 运行环境

- Python 3.x
- NumPy, SciPy, Matplotlib
- 建议 GPU 环境（部分算法需要）

## 说明

本作业为学习性质的课程项目，实验效果不理想，仅供参考。

---

**作者**: 邓嘉轩
**课程**: SAR 相关课程
**状态**: 已终止/不再继续研究
