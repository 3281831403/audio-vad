# audio-vad

## 题面

给会议录音加自动切句后，静音段被当成说话、句子还总被切碎：帧能量按固定 20 ms 计算，10 ms 帧长时结果错位；能量用线性值而不是 dB，阈值不可比；过零率按整段算，不按帧；判决用单阈值，噪声边缘反复翻转；滞回只对能量生效，ZCR 仍抖；平滑不做最短静音合并，句子被切碎；区间转采样点用浮点累加，长音频漂移 10 ms；尾部保持不设上限，静音段被无限延长；批量遇坏轨只跳过，输出缺帧；静音段（全零）过零率为 0 被判成语音。vad/detector.py 是核心，tests/test_detector.py 15个红。要修的是：…

## 运行方式

在 Windows PowerShell 中执行：

```powershell
cd D:\gsb-20261119\audio-vad
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，并且没有 collection error、import error 或 syntax error；15 条失败都来自桩函数抛出的 `NotImplementedError`。实现 `vad/detector.py` 后应全部通过。帧长、hop、采样率和三段阈值都由调用方显式传入；测试不使用随机数、不 sleep、不联网，只依赖标准库与 pytest。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_detector.py` 不可修改，也不可新增、删除或跳过测试。修复应发生在 `vad/detector.py`；保持 8 个公开函数的名字和签名不变，异常统一使用 `VadError`。
