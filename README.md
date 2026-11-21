# render-color-grading

## 题面

套用电影感 LUT 后，画面出现色阶、高光还回绕成黑点：LUT 采样用 `i / size` 当步长，最后一格落在 `(size-1)/size`，1.0 端点永远采不到；三线性插值退化成八个角逐个取最近，平滑渐变出现台阶；查表不做 clamp，越界坐标变成负索引去取远端颜色；`gamma` 为 0 不校验，`base ** (1.0 / gamma)` 直接除零；饱和度拿 RGB 算术均值当亮度，绿色和红色一样亮；对比度支点写死 0.5，暗部被压死、亮部被拉爆；LUT 尺寸不校验，非立方或只有一格时插值直接越界；调色结果不钳制到 0..1，高光溢出后回绕成黑点；批量调色多图共用同一块输出缓冲，图与图之间串色；`amount < 0` 不钳制，负饱和度把颜色反相。grading/lut.py 是核心，tests/test_lut.py 15个红。要修的是：采样端点、三线性插值、坐标越界 clamp、gamma 校验、Rec.709 亮度权重、可配置对比度支点、LUT 尺寸校验、输出钳制、批量缓冲隔离、负 amount 钳制——十项全部落在 `grading/lut.py` 的 8 个桩函数里。

## 运行方式

在仓库根目录（`render-color-grading/`）执行：

```
python -m pytest tests -q
```

初始（失败）状态必须恰好是 15 个失败、0 个 error、0 个跳过：

```
................
15 failed in 0.10s
```

（`15 failed` 是硬指标，耗时数字随机器不同；只要失败数正好是 15、且没有 error/collection error 即可。修复完成后目标是 `15 passed`。）

修复约束：

- 只改 `grading/lut.py`，可自由添加私有辅助函数。
- 只用 Python 标准库（`math`）与 `pytest`，离线运行、不使用随机数。
- 数值比较容差为 `1e-6`，颜色通道一律是 `[0.0, 1.0]` 的浮点数。
- 图像是 `list[list[tuple[float, float, float]]]`，LUT 是 `list[list[list[tuple[float, float, float]]]]`（`lut[r][g][b]`）。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 注意

`tests/test_lut.py` 不可修改。15 个测试就是验收标准：10 个测试各对应一个缺陷，另外 5 个覆盖恒等 LUT 往返、`saturation=0` 灰度、`size=1` 抛 `GradeError`、空图像、批量顺序保持。修改测试文件视为无效提交。

