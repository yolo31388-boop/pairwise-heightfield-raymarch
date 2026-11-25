# pairwise-heightfield-raymarch

## 领域

游戏、图形与多媒体

## 主题

高度场光线步进

## 说明

自适应步长、LOD 一致求交与上传栅栏。

初始状态下 `graphics/heightfield_march.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
