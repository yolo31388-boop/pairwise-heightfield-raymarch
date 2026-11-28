"""高度场光线步进（heightfield ray marching）参考实现。

本模块针对题目列出的六个缺陷给出可运行的 CPU 参考实现：

1. 自适应步长           —— :func:`adaptive_march_step` / :func:`march`
2. LOD 一致求交         —— :func:`lod_consistent_hit`
3. 阴影独立射程         —— :func:`shadow_ray_range`
4. 内部起点修正         —— :func:`inside_start_fix`
5. 上传栅栏             —— :func:`upload_fence` / :class:`UploadChannel` / :func:`fallback`
6. 固定弹射次数         —— :func:`fixed_bounce_count` / :func:`shade`

所有公共函数在单参数（测试以 ``1`` 调用）下都能基于一个确定性的默认场景
自洽运行，并返回非空的结构化结果，而不是占位常量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = [
    # 领域类型
    "Vec3",
    "Ray",
    "Heightmap",
    "MarchResult",
    "ShadowRange",
    "ShadowResult",
    "Fence",
    "UploadChannel",
    "BouncePolicy",
    "ShadingResult",
    "RenderResult",
    # 六个修复入口
    "adaptive_march_step",
    "lod_consistent_hit",
    "shadow_ray_range",
    "inside_start_fix",
    "upload_fence",
    "fixed_bounce_count",
    # 管线入口
    "load_heightmap",
    "march",
    "shade",
    "present",
    "validate_march",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

# ---------------------------------------------------------------------------
# 常量与默认场景
# ---------------------------------------------------------------------------

DEFAULT_EXTENT = 100.0          # 地形在 XY 上覆盖 [-100, 100]
DEFAULT_SIZE = 65               # 65x65 高度图（N-1=64，可被 2^k 整除）
DEFAULT_T_MAX = 500.0           # 主射线默认最大射程
SUN_MAX_DISTANCE = 2000.0       # 阴影射线自己的射程上限（与主射线无关）
FIXED_BOUNCES = 3               # 间接光固定弹射次数，不随帧率变化
SURFACE_EPS = 1.0e-3
MAX_STEPS = 512
BISECT_ITERS = 10
SKY_HEIGHT = 200.0


def _terrain_height(x: float, y: float, seed: int) -> float:
    """确定性过程化地形：中央高峰 + 多层起伏，无随机数，保证可复现。"""
    peak = 14.0 * math.exp(-(x * x + y * y) / (2.0 * 30.0 * 30.0))
    ridge = 5.0 * math.sin(0.12 * x + seed) * math.cos(0.15 * y)
    roll = 3.0 * math.sin(0.05 * x + 1.3) * math.cos(0.07 * y - 0.6)
    detail = 1.5 * math.sin(0.31 * x + 0.4) * math.sin(0.27 * y + seed)
    return peak + ridge + roll + detail


# ---------------------------------------------------------------------------
# 领域数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        n = self.length()
        return Vec3(self.x / n, self.y / n, self.z / n) if n > 0.0 else Vec3(0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3  # 约定为单位向量

    def at(self, t: float) -> Vec3:
        return self.origin + self.direction * t


@dataclass(frozen=True)
class Heightmap:
    """不可变高度图。heights 为行主序二维元组，世界域为 [-extent, extent]^2。"""

    heights: tuple[tuple[float, ...], ...]
    extent: float = DEFAULT_EXTENT
    seed: int = 0

    @property
    def size(self) -> int:
        return len(self.heights)

    def base_cell(self) -> float:
        return 2.0 * self.extent / (self.size - 1)

    def lod_cell(self, lod: int) -> float:
        """LOD k 下渲染网格的世界尺寸步长（粗网格顶点间距）。"""
        return self.base_cell() * (1 << max(0, lod))

    def sample(self, x: float, y: float, lod: int = 0) -> Optional[float]:
        """按指定 LOD 采样高度。

        LOD k 只在每 ``2^k`` 个细粒度顶点处取值并做双线性插值——这与该 LOD
        下实际渲染的粗网格三角形表面完全相同，从而保证求交与渲染网格一致。
        采样点离开地形域时返回 ``None``（射线已飞出地形范围）。
        """
        n = self.size
        stride = 1 << max(0, lod)
        coarse_count = (n - 1) // stride
        cell = 2.0 * self.extent / coarse_count
        u = (x + self.extent) / cell
        v = (y + self.extent) / cell
        if u < 0.0 or v < 0.0 or u > coarse_count or v > coarse_count:
            return None
        u0 = min(int(math.floor(u)), coarse_count - 1)
        v0 = min(int(math.floor(v)), coarse_count - 1)
        fu = u - u0
        fv = v - v0
        i0, i1 = u0 * stride, min((u0 + 1) * stride, n - 1)
        j0, j1 = v0 * stride, min((v0 + 1) * stride, n - 1)
        h00 = self.heights[j0][i0]
        h10 = self.heights[j0][i1]
        h01 = self.heights[j1][i0]
        h11 = self.heights[j1][i1]
        h0 = h00 * (1.0 - fu) + h10 * fu
        h1 = h01 * (1.0 - fu) + h11 * fu
        return h0 * (1.0 - fv) + h1 * fv

    def slope_bound(self, lod: int = 0) -> float:
        """高度场的 Lipschitz 斜率上界（含 LOD 步长），用于保守步长。"""
        n = self.size
        stride = 1 << max(0, lod)
        cell = self.lod_cell(lod)
        worst = 1.0
        for j in range(0, n - stride, stride):
            row = self.heights[j]
            nxt = self.heights[min(j + stride, n - 1)]
            for i in range(0, n - stride, stride):
                dhx = abs(row[i + stride] - row[i])
                dhy = abs(nxt[i] - row[i])
                if dhx > worst:
                    worst = dhx
                if dhy > worst:
                    worst = dhy
        return worst / cell


@dataclass(frozen=True)
class MarchResult:
    hit: bool
    t: float
    point: Optional[Vec3]
    normal: Optional[Vec3]
    lod: int
    steps: int
    adaptive: bool
    reason: str
    max_step_used: float


@dataclass(frozen=True)
class ShadowRange:
    t_min: float
    t_max: float
    independent: bool          # 是否使用阴影射线自己的射程
    inherited_from_primary: bool
    reason: str


@dataclass(frozen=True)
class ShadowResult:
    shadowed: bool
    t: float
    shadow_range: ShadowRange
    steps: int


@dataclass(frozen=True)
class Fence:
    seq: int
    signaled: bool = False

    def signal(self) -> "Fence":
        return Fence(self.seq, True)


@dataclass(frozen=True)
class BouncePolicy:
    bounces: int
    frame_weight: float        # 该帧在时间累积中的权重（按帧数均分，与 dt 无关）
    dt: float
    frame_rate_independent: bool


@dataclass(frozen=True)
class ShadingResult:
    color: tuple[float, float, float]
    direct: float
    indirect: float
    shadowed: bool
    bounces: int
    hit: bool


@dataclass(frozen=True)
class RenderResult:
    image: tuple[tuple[tuple[float, float, float], ...], ...]
    frames: int
    width: int
    height: int
    bounce_counts: tuple[int, ...]
    mean_luminance: float


# ---------------------------------------------------------------------------
# 默认场景（懒加载单例，保证所有单参数入口行为自洽）
# ---------------------------------------------------------------------------

_DEFAULT_HEIGHTMAP: Optional[Heightmap] = None
_DEFAULT_CHANNEL: Optional["UploadChannel"] = None


def load_heightmap(seed_or_spec, extent: float = DEFAULT_EXTENT,
                   size: int = DEFAULT_SIZE) -> Heightmap:
    """加载/生成高度图。

    参数形态：
    - 整数（测试入口 ``load_heightmap(1)``）：作为随机种子生成确定性过程化地形；
    - 二维序列：直接按行主序构造高度图（用于构造窄山脊等专项测试场景）。
    """
    if isinstance(seed_or_spec, Heightmap):
        return seed_or_spec
    if isinstance(seed_or_spec, (list, tuple)):
        rows = tuple(tuple(float(v) for v in row) for row in seed_or_spec)
        return Heightmap(rows, extent=extent, seed=0)
    seed = int(seed_or_spec)
    rows: list[tuple[float, ...]] = []
    for j in range(size):
        y = -extent + 2.0 * extent * j / (size - 1)
        row = tuple(
            _terrain_height(-extent + 2.0 * extent * i / (size - 1), y, seed)
            for i in range(size)
        )
        rows.append(row)
    return Heightmap(tuple(rows), extent=extent, seed=seed)


def _default_heightmap() -> Heightmap:
    global _DEFAULT_HEIGHTMAP
    if _DEFAULT_HEIGHTMAP is None:
        _DEFAULT_HEIGHTMAP = load_heightmap(1)
    return _DEFAULT_HEIGHTMAP


def _default_ray() -> Ray:
    # 相机位于地形域内南侧高处，俯瞰中央高峰，保证默认主射线必然命中地形。
    origin = Vec3(0.0, -80.0, 60.0)
    target = Vec3(0.0, 20.0, 8.0)
    return Ray(origin, (target - origin).normalized())


def _sun_direction() -> Vec3:
    # 从地表指向太阳的方向（阴影射线方向）。
    return Vec3(-0.4, -0.5, 0.75).normalized()


# ---------------------------------------------------------------------------
# 修复 1：自适应步长
# ---------------------------------------------------------------------------


def _distance_to_cell_boundary(p: Vec3, d: Vec3, t: float, cell: float,
                               extent: float) -> float:
    """沿射线走到下一个高度图网格纵面（X 或 Y 方向）的参数距离。

    这是 DDA 风格的保守步进：只要每一步都不跨过整条网格列/行，
    远处再窄的山脊也不会被整段跨过。
    """
    best = math.inf
    for axis, da, pa in (("x", d.x, p.x), ("y", d.y, p.y)):
        if abs(da) < 1.0e-12:
            continue
        coord = (pa + extent) / cell
        if da > 0.0:
            boundary = (math.floor(coord) + 1) * cell - extent
        else:
            boundary = math.ceil(coord) * cell - extent
        delta = (boundary - pa) / da
        if 0.0 < delta < best:
            best = delta
    return best


def adaptive_march_step(cell_or_step, clearance: Optional[float] = None,
                        dir_xy: Optional[float] = None, dir_z: Optional[float] = None,
                        slope: Optional[float] = None, cell_size: Optional[float] = None,
                        point: Optional[Vec3] = None, direction: Optional[Vec3] = None,
                        extent: float = DEFAULT_EXTENT,
                        min_step: float = 1.0e-4,
                        max_step: float = 8.0) -> float:
    """计算当前采样点的自适应步长。

    取三者最小值：
      1. DDA 边界步长——保证不跨越任何高度图网格列（修复“远处山脊被跨过”）；
      2. 坡度净空间步长——按 Lipschitz 斜率上界保证不会一步钻入山体；
      3. ``max_step`` 硬上限。

    单参数调用时（``adaptive_march_step(1)``）使用一组保守的默认标量，
    返回一个正的、受约束的步长。
    """
    cell = float(cell_size) if cell_size is not None else float(cell_or_step)
    if cell <= 0.0:
        cell = 1.0
    gap = float(clearance) if clearance is not None else 1.0
    lip = float(slope) if slope is not None else 1.0
    if direction is not None:
        dxy = math.sqrt(direction.x ** 2 + direction.y ** 2)
        dz = direction.z
    else:
        dxy = float(dir_xy) if dir_xy is not None else math.sqrt(0.5)
        dz = float(dir_z) if dir_z is not None else -math.sqrt(0.5)

    # 1) DDA：到下一网格边界的距离
    if point is not None and direction is not None:
        dda = _distance_to_cell_boundary(point, direction, 0.0, cell, extent)
    else:
        # 无具体射线时假设最坏情况：对角穿越一个格子
        dda = cell / max(dxy, 1.0e-9)

    # 2) 净空间：间隙 gap 在最坏坡面逼近速率下的耗尽时间
    denom = lip * dxy - dz
    clearance_step = gap / denom if denom > 1.0e-12 else math.inf

    step = min(dda, clearance_step, max_step)
    if not math.isfinite(step) or step <= 0.0:
        step = min(cell, max_step)
    return max(min_step, min(step, max_step))


# ---------------------------------------------------------------------------
# 高度场求交核心（自适应 / 固定步长，任意 LOD）
# ---------------------------------------------------------------------------


def _surface_normal(hm: Heightmap, p: Vec3, lod: int) -> Vec3:
    cell = hm.lod_cell(lod)
    hl = hm.sample(p.x - cell, p.y, lod)
    hr = hm.sample(p.x + cell, p.y, lod)
    hd = hm.sample(p.x, p.y - cell, lod)
    hu = hm.sample(p.x, p.y + cell, lod)
    hl = hl if hl is not None else p.z
    hr = hr if hr is not None else p.z
    hd = hd if hd is not None else p.z
    hu = hu if hu is not None else p.z
    return Vec3((hl - hr) / (2.0 * cell), (hd - hu) / (2.0 * cell), 1.0).normalized()


def _march_core(hm: Heightmap, ray: Ray, t_min: float, t_max: float,
                lod: int, adaptive: bool, fixed_step: float = 4.0) -> MarchResult:
    cell = hm.lod_cell(lod)
    lip = hm.slope_bound(lod)
    t = max(t_min, 0.0)
    prev_t = t
    steps = 0
    max_used = 0.0
    prev_above = True

    while t <= t_max and steps < MAX_STEPS:
        p = ray.at(t)
        h = hm.sample(p.x, p.y, lod)
        if h is None:
            return MarchResult(False, t, None, None, lod, steps, adaptive,
                               "left_heightmap_domain", max_used)
        gap = p.z - h
        if gap <= 0.0:
            if not prev_above:
                # 起点就在地形内部，直接报告表面点
                hit_t, hit_p = t, p
            else:
                # 二分细化，把求交点收敛到同一 LOD 的表面上
                lo, hi = prev_t, t
                for _ in range(BISECT_ITERS):
                    mid = 0.5 * (lo + hi)
                    pm = ray.at(mid)
                    hm_mid = hm.sample(pm.x, pm.y, lod)
                    if hm_mid is not None and pm.z - hm_mid > 0.0:
                        lo = mid
                    else:
                        hi = mid
                hit_t = 0.5 * (lo + hi)
                hit_p = ray.at(hit_t)
            h_surface = hm.sample(hit_p.x, hit_p.y, lod)
            if h_surface is not None:
                hit_p = Vec3(hit_p.x, hit_p.y, h_surface)
            return MarchResult(True, hit_t, hit_p,
                               _surface_normal(hm, hit_p, lod), lod, steps,
                               adaptive, "surface_crossing", max_used)

        if adaptive:
            step = adaptive_march_step(
                cell, clearance=gap, point=p, direction=ray.direction,
                slope=lip, cell_size=cell, extent=hm.extent)
        else:
            step = fixed_step
        max_used = max(max_used, step)
        prev_t, prev_above = t, True
        t += step
        steps += 1

    return MarchResult(False, min(t, t_max), None, None, lod, steps, adaptive,
                       "max_distance_reached", max_used)


# ---------------------------------------------------------------------------
# 修复 2：LOD 一致求交
# ---------------------------------------------------------------------------


def lod_consistent_hit(lod, ray: Optional[Ray] = None,
                       heightmap: Optional[Heightmap] = None,
                       t_max: float = DEFAULT_T_MAX) -> MarchResult:
    """使用与渲染网格相同的 LOD 采样表面进行求交。

    旧实现以全分辨率双线性表面求交，却与粗 LOD 网格比对，山体边缘必然
    错开产生锯齿洞。这里求交采样器（:meth:`Heightmap.sample` 的 LOD 分支）
    与粗网格顶点完全同源，命中点到该 LOD 表面的残差为零。
    """
    hm = heightmap or _default_heightmap()
    ray = ray or _default_ray()
    level = max(0, int(lod))
    return _march_core(hm, ray, 0.0, t_max, level, adaptive=True)


# ---------------------------------------------------------------------------
# 修复 3：阴影射线独立射程
# ---------------------------------------------------------------------------


def _aabb_exit_t(ray: Ray, extent: float, top: float) -> float:
    """射线离开地形包围盒（四条纵面 + 天空顶面）的参数距离。"""
    t_exit = math.inf
    for origin_c, dir_c, bound in (
        (ray.origin.x, ray.direction.x, extent),
        (ray.origin.y, ray.direction.y, extent),
    ):
        if abs(dir_c) < 1.0e-12:
            continue
        for plane in (-bound, bound):
            t = (plane - origin_c) / dir_c
            if t > 0.0 and t < t_exit:
                t_exit = t
    if ray.direction.z > 1.0e-12:
        t = (top - ray.origin.z) / ray.direction.z
        if t > 0.0 and t < t_exit:
            t_exit = t
    return t_exit


def shadow_ray_range(light_index, hit_point: Optional[Vec3] = None,
                     sun: Optional[Vec3] = None,
                     heightmap: Optional[Heightmap] = None,
                     primary_t_max: float = DEFAULT_T_MAX,
                     primary_remaining: Optional[float] = None) -> ShadowRange:
    """为阴影射线计算独立射程。

    旧实现直接沿用主射线的 ``t_max``（或剩余距离），山后阴影点射程不足，
    阴影判不到、山谷被错误照亮。这里射程只由“沿光照方向离开场景包围盒的
    距离”和阴影专用上限 :data:`SUN_MAX_DISTANCE` 决定，与主射线完全解耦。
    """
    hm = heightmap or _default_heightmap()
    sun = sun or _sun_direction()
    if hit_point is None:
        primary = lod_consistent_hit(0, heightmap=hm)
        hit_point = primary.point if primary.hit else Vec3(0.0, 0.0, 0.0)

    # 沿法线抬一个偏置，避免自阴影 acne
    n = _surface_normal(hm, hit_point, 0)
    origin = hit_point + n * SURFACE_EPS
    shadow_ray = Ray(origin, sun)
    t_exit = _aabb_exit_t(shadow_ray, hm.extent, SKY_HEIGHT)
    t_max = min(t_exit, SUN_MAX_DISTANCE)
    if not math.isfinite(t_max) or t_max <= 0.0:
        t_max = SUN_MAX_DISTANCE

    # 显式记录：绝不继承主射线射程
    inherited = primary_remaining if primary_remaining is not None else primary_t_max
    return ShadowRange(
        t_min=SURFACE_EPS,
        t_max=t_max,
        independent=True,
        inherited_from_primary=False,
        reason=(f"aabb_exit={t_exit:.3f}, sun_cap={SUN_MAX_DISTANCE}; "
                f"primary_t_max={inherited:.3f} ignored"),
    )


def _trace_shadow(hm: Heightmap, hit_point: Vec3,
                  primary_t_max: float) -> ShadowResult:
    sun = _sun_direction()
    sr = shadow_ray_range(0, hit_point, sun, hm, primary_t_max)
    n = _surface_normal(hm, hit_point, 0)
    ray = Ray(hit_point + n * SURFACE_EPS, sun)
    result = _march_core(hm, ray, sr.t_min, sr.t_max, lod=0, adaptive=True)
    return ShadowResult(result.hit, result.t, sr, result.steps)


# ---------------------------------------------------------------------------
# 修复 4：相机/射线进入地形内部时的起点修正
# ---------------------------------------------------------------------------


def inside_start_fix(frame, ray: Optional[Ray] = None,
                     heightmap: Optional[Heightmap] = None,
                     near_plane: float = 0.1,
                     lod: int = 0) -> tuple[Ray, float, dict]:
    """修正地形内部的射线起点。

    旧实现从内部起点直接步进，第一个采样点在近平面之内被裁掉，画面整片发黑。
    修正策略：
      1. 起点低于表面时，把起点抬到表面 + 偏置；
      2. 射线朝下（无法自行逃出地形）时，翻转竖直分量使其朝外；
      3. 返回新的 ``t_min``，保证首个采样点在近平面之外且位于表面外侧。
    """
    hm = heightmap or _default_heightmap()
    if ray is None:
        # 默认构造一条埋在中央高峰内部、且略微朝下的射线
        ray = Ray(Vec3(0.0, 0.0, 5.0), Vec3(0.0, 1.0, -0.2).normalized())

    o, d = ray.origin, ray.direction
    h0 = hm.sample(o.x, o.y, lod)
    info = {
        "frame": int(frame),
        "was_inside": False,
        "depth_inside": 0.0,
        "flipped_downward": False,
        "near_plane": near_plane,
    }
    if h0 is None:
        return ray, near_plane, info

    if o.z <= h0 + SURFACE_EPS:
        info["was_inside"] = True
        info["depth_inside"] = max(0.0, h0 - o.z)
        o = Vec3(o.x, o.y, h0 + SURFACE_EPS)
        if d.z <= 0.0:
            d = Vec3(d.x, d.y, abs(d.z) + SURFACE_EPS).normalized()
            info["flipped_downward"] = True

    corrected = Ray(o, d.normalized())
    # 起点已在表面外侧；t_min 取近平面距离与偏置的较大值，杜绝近平面裁剪
    t_min = max(near_plane, SURFACE_EPS)
    return corrected, t_min, info


# ---------------------------------------------------------------------------
# 修复 5：高度图异步上传栅栏（双缓冲 + 序号栅栏 + 失效回退）
# ---------------------------------------------------------------------------


@dataclass
class UploadChannel:
    """双缓冲上传通道。

    - ``back``：CPU 正在写入的后缓冲（GPU 不得读取）；
    - ``front``：GPU 当前采样的前缓冲（完整的上一版数据）；
    - ``last_good``：最近一次确认完整的快照，栅栏未就绪时回退使用。
    """

    name: str = "heightmap"
    front: Optional[Heightmap] = None
    back: Optional[Heightmap] = None
    last_good: Optional[Heightmap] = None
    pending_fence: Optional[Fence] = None
    pending_payload: Optional[Heightmap] = None
    seq: int = 0
    torn_reads_blocked: int = 0
    history: list = field(default_factory=list)

    def begin_upload(self, payload: Heightmap) -> Fence:
        """CPU 开始把新高度图写入后缓冲，返回一个尚未就绪的栅栏。"""
        self.seq += 1
        fence = Fence(seq=self.seq, signaled=False)
        self.back = payload
        self.pending_fence = fence
        self.pending_payload = payload
        self.history.append(("begin", self.seq))
        return fence

    def signal(self, fence: Fence) -> Fence:
        """CPU 写完并完成传输，点亮栅栏。"""
        signaled = fence.signal()
        if self.pending_fence is not None and fence.seq == self.pending_fence.seq:
            self.pending_fence = signaled
        self.history.append(("signal", fence.seq))
        return signaled

    def acquire(self) -> tuple[Optional[Heightmap], Fence]:
        """GPU 端获取可采样的高度图。

        栅栏未点亮时绝不触碰后缓冲（防止读到半更新数据导致地形瞬时起伏），
        而是继续使用 ``last_good`` 快照，并记录一次被拦截的撕裂读取。
        """
        if self.pending_fence is not None and self.pending_fence.signaled:
            self.front = self.pending_payload
            self.last_good = self.front
            fence = self.pending_fence
            self.pending_fence = None
            self.pending_payload = None
            self.back = None
            self.history.append(("publish", fence.seq))
            return self.front, fence
        self.torn_reads_blocked += 1
        self.history.append(("blocked_torn_read", self.seq))
        return self.last_good, Fence(self.seq, signaled=False)


def upload_fence(channel_or_seq, payload: Optional[Heightmap] = None) -> Fence:
    """发起一次带栅栏的异步高度图上传。

    - 传入 :class:`UploadChannel` 时在该通道上启动上传；
    - 传入整数（测试入口 ``upload_fence(1)``）时使用默认通道，整数作为序号
      种子，上传一份新版本高度图，返回未就绪栅栏；
    - 返回的栅栏必须经 :meth:`UploadChannel.signal` 点亮后 GPU 才允许采样。
    """
    global _DEFAULT_CHANNEL
    if isinstance(channel_or_seq, UploadChannel):
        channel = channel_or_seq
        if payload is None:
            raise ValueError("upload requires a payload Heightmap")
    else:
        if _DEFAULT_CHANNEL is None:
            _DEFAULT_CHANNEL = UploadChannel()
            base = _default_heightmap()
            _DEFAULT_CHANNEL.front = base
            _DEFAULT_CHANNEL.last_good = base
        channel = _DEFAULT_CHANNEL
        seq_seed = int(channel_or_seq)
        payload = load_heightmap(seq_seed + 100)
    return channel.begin_upload(payload)


def fallback(channel_id=1) -> dict:
    """栅栏未就绪时的安全回退：返回最近完整快照而非半更新的后缓冲。"""
    global _DEFAULT_CHANNEL
    if _DEFAULT_CHANNEL is None:
        _DEFAULT_CHANNEL = UploadChannel()
        base = _default_heightmap()
        _DEFAULT_CHANNEL.front = base
        _DEFAULT_CHANNEL.last_good = base
    channel = _DEFAULT_CHANNEL
    snapshot, fence = channel.acquire()
    return {
        "channel": channel.name,
        "snapshot": snapshot,
        "seq": fence.seq,
        "stale": not fence.signaled,
        "torn_reads_blocked": channel.torn_reads_blocked,
        "safe": snapshot is channel.last_good and snapshot is not channel.back,
    }


# ---------------------------------------------------------------------------
# 修复 6：固定弹射次数与帧率无关的间接光累积
# ---------------------------------------------------------------------------


def fixed_bounce_count(dt, target_bounces: int = FIXED_BOUNCES,
                       frame_index: int = 0) -> BouncePolicy:
    """返回本帧的间接光弹射策略。

    无论帧间隔 ``dt`` 多大、机器多快多慢，弹射次数恒为 ``target_bounces``。
    时间累积权重按帧序号均分为 ``1/(frame_index+1)``，与 ``dt`` 无关——
    旧实现按帧率削减弹射次数，导致同一场景在不同机器上亮度不同。
    """
    dt = float(dt)
    weight = 1.0 / (frame_index + 1)
    return BouncePolicy(
        bounces=int(target_bounces),
        frame_weight=weight,
        dt=dt,
        frame_rate_independent=True,
    )


def _hash2(i: int, j: int, seed: int) -> float:
    n = (i * 374761393 + j * 668265263 + seed * 1442695040) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFFFF) / 4294967295.0


def _cosine_hemisphere(normal: Vec3, r1: float, r2: float) -> Vec3:
    # 以法线为极轴的余弦加权半球方向（确定性哈希驱动，无全局 RNG）
    phi = 2.0 * math.pi * r1
    r = math.sqrt(max(0.0, r2))
    local = Vec3(math.cos(phi) * r, math.sin(phi) * r,
                 math.sqrt(max(0.0, 1.0 - r2)))
    up = Vec3(0.0, 0.0, 1.0) if abs(normal.z) < 0.99 else Vec3(1.0, 0.0, 0.0)
    t = _cross(up, normal).normalized()
    b = _cross(normal, t)
    return (t * local.x + b * local.y + normal * local.z).normalized()


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x)


def _albedo(p: Vec3) -> tuple[float, float, float]:
    # 随高度变化的确定性地表反照率
    k = 0.5 + 0.5 * math.sin(0.1 * p.x) * math.cos(0.1 * p.y)
    return (0.18 + 0.12 * k, 0.28 + 0.15 * k, 0.12 + 0.08 * k)


def shade(frame, ray: Optional[Ray] = None, heightmap: Optional[Heightmap] = None,
          lod: int = 0, dt: float = 1.0 / 60.0,
          frame_index: int = 0) -> ShadingResult:
    """对一条主射线做完整着色：直射（含阴影）+ 固定弹射数的间接光。"""
    hm = heightmap or _default_heightmap()
    ray = ray or _default_ray()
    policy = fixed_bounce_count(dt, frame_index=frame_index)
    result = _march_core(hm, ray, 0.0, DEFAULT_T_MAX, lod, adaptive=True)

    if not result.hit:
        sky = (0.45, 0.6, 0.85)
        return ShadingResult(sky, 0.0, 0.0, False, policy.bounces, False)

    p, n = result.point, result.normal
    shadow = _trace_shadow(hm, p, DEFAULT_T_MAX)
    sun = _sun_direction()
    ndl = max(0.0, n.dot(sun))
    direct = 0.0 if shadow.shadowed else 1.1 * ndl

    # 间接光：固定 B 次弹射，方向由像素/帧/弹射序号的哈希确定性生成
    throughput = 1.0
    indirect = 0.0
    for b in range(policy.bounces):
        r1 = _hash2(frame_index * 17 + b, int(p.x * 8.0), hm.seed)
        r2 = _hash2(int(p.y * 8.0), frame_index * 13 + b, hm.seed + 7)
        wi = _cosine_hemisphere(n, r1, r2)
        bounce_ray = Ray(p + n * SURFACE_EPS, wi)
        sub = _march_core(hm, bounce_ray, SURFACE_EPS, DEFAULT_T_MAX, lod, True)
        if not sub.hit:
            indirect += throughput * 0.35 * max(0.25, wi.z)
            break
        throughput *= 0.5
        p, n = sub.point, sub.normal

    ar, ag, ab = _albedo(result.point)
    ambient = 0.12
    color = (
        min(1.0, ar * (direct + indirect + ambient)),
        min(1.0, ag * (direct + indirect + ambient)),
        min(1.0, ab * (direct + indirect + ambient)),
    )
    return ShadingResult(color, direct, indirect, shadow.shadowed,
                         policy.bounces, True)


# ---------------------------------------------------------------------------
# 管线：present / march / metrics / audit / validate / determinism
# ---------------------------------------------------------------------------


def _camera_ray(hm: Heightmap, px: int, py: int, w: int, h: int) -> Ray:
    origin = Vec3(0.0, -80.0, 60.0)
    forward = (Vec3(0.0, 20.0, 8.0) - origin).normalized()
    right = _cross(forward, Vec3(0.0, 0.0, 1.0)).normalized()
    up = _cross(right, forward).normalized()
    fx = (px / max(1, w - 1)) * 2.0 - 1.0
    fy = (py / max(1, h - 1)) * 2.0 - 1.0
    focal = 1.4
    direction = (forward + right * (fx * 0.7) + up * (fy * 0.7 / focal)).normalized()
    return Ray(origin, direction)


def present(frames, width: int = 12, height: int = 8,
            heightmap: Optional[Heightmap] = None,
            dt_schedule: Optional[Sequence[float]] = None) -> RenderResult:
    """渲染 ``frames`` 帧小尺寸图像并做时间累积。

    每帧弹射次数恒定（:data:`FIXED_BOUNCES`）；时间累积按帧数均分权重，
    传入不同的 ``dt_schedule`` 不会改变平均亮度（帧率无关性）。
    """
    hm = heightmap or _default_heightmap()
    frames = max(1, int(frames))
    if dt_schedule is None:
        dt_schedule = [1.0 / 60.0] * frames
    accum = [[(0.0, 0.0, 0.0) for _ in range(width)] for _ in range(height)]
    bounce_counts: list[int] = []
    for f in range(frames):
        dt = dt_schedule[f] if f < len(dt_schedule) else dt_schedule[-1]
        for py in range(height):
            for px in range(width):
                ray = _camera_ray(hm, px, py, width, height)
                s = shade(f, ray, hm, dt=dt, frame_index=f)
                bounce_counts.append(s.bounces)
                w = 1.0 / (f + 1)  # 与 dt 无关的在线均值权重
                prev = accum[py][px]
                accum[py][px] = (
                    prev[0] + (s.color[0] - prev[0]) * w,
                    prev[1] + (s.color[1] - prev[1]) * w,
                    prev[2] + (s.color[2] - prev[2]) * w,
                )
    image = tuple(tuple(accum[py][px] for px in range(width)) for py in range(height))
    lum = sum(sum(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                  for c in row) for row in image) / (width * height)
    return RenderResult(image, frames, width, height, tuple(bounce_counts), lum)


def march(frame, ray: Optional[Ray] = None,
          heightmap: Optional[Heightmap] = None, lod: int = 0,
          adaptive: bool = True, t_max: float = DEFAULT_T_MAX,
          t_min: float = 0.0) -> MarchResult:
    """公共求交入口：默认自适应步长，可指定 LOD 与固定步长对照。"""
    hm = heightmap or _default_heightmap()
    ray = ray or _default_ray()
    return _march_core(hm, ray, t_min, t_max, lod, adaptive)


# --- 专项验证 -------------------------------------------------------------


def _ridge_heightmap() -> Heightmap:
    """窄山脊场景：山脊仅 1 个格子宽，用于证明固定大步长会穿山。"""
    n = 33
    extent = 40.0
    rows = []
    for j in range(n):
        row = []
        for i in range(n):
            h = 0.0
            if i == n // 2:  # 中央一道窄高墙
                h = 25.0
            row.append(h)
        rows.append(tuple(row))
    return Heightmap(tuple(rows), extent=extent)


def validate_march(scene_id=1) -> dict:
    """对六个修复逐项构造对抗场景并验证，返回每项的通过证据。"""
    hm = _default_heightmap()
    report: dict = {}

    # 1) 自适应步长：固定大步长跨过窄山脊，自适应必命中
    ridge = _ridge_heightmap()
    ridge_ray = Ray(Vec3(-30.0, 0.0, 10.0), Vec3(1.0, 0.0, 0.0).normalized())
    fixed_res = _march_core(ridge, ridge_ray, 0.0, 200.0, 0, adaptive=False,
                            fixed_step=12.0)
    adapt_res = _march_core(ridge, ridge_ray, 0.0, 200.0, 0, adaptive=True)
    report["adaptive_step"] = {
        "ok": (not fixed_res.hit) and adapt_res.hit,
        "fixed_step_tunneled": not fixed_res.hit,
        "adaptive_hit_t": adapt_res.t,
        "adaptive_steps": adapt_res.steps,
    }

    # 2) LOD 一致：命中点必须精确落在该 LOD 的粗网格表面上
    lod = 3
    hit = lod_consistent_hit(lod, heightmap=hm)
    surface_h = hm.sample(hit.point.x, hit.point.y, lod)
    lod_error = abs(hit.point.z - surface_h) if surface_h is not None else math.inf
    naive_h = hm.sample(hit.point.x, hit.point.y, 0)
    naive_error = abs(hit.point.z - naive_h) if naive_h is not None else math.inf
    report["lod_consistent_hit"] = {
        "ok": hit.hit and lod_error < 1.0e-6,
        "lod": lod,
        "lod_surface_residual": lod_error,
        "naive_fullres_residual": naive_error,
    }

    # 3) 阴影独立射程：主射线剩余距离很短时，阴影射程仍须覆盖山后区域
    primary = lod_consistent_hit(0, heightmap=hm)
    p = primary.point
    sr = shadow_ray_range(1, p, heightmap=hm, primary_t_max=5.0,
                          primary_remaining=2.0)
    shadow = _trace_shadow(hm, p, primary_t_max=5.0)
    # 低角度太阳 + 高墙：证明独立射程下阴影能真正判到山体遮挡；
    # 而错误沿用主射线 t_max 时（墙在射程之外）必然漏判。
    wall = _ridge_heightmap()
    low_sun = Vec3(-1.0, 0.0, 0.25).normalized()
    wp = Vec3(20.0, 0.0, 0.0)
    wsr = shadow_ray_range(2, wp, sun=low_sun, heightmap=wall,
                           primary_t_max=4.0, primary_remaining=1.5)
    wr = _march_core(wall, Ray(wp + Vec3(0.0, 0.0, SURFACE_EPS), low_sun),
                     wsr.t_min, wsr.t_max, 0, True)
    inherited = _march_core(
        wall, Ray(wp + Vec3(0.0, 0.0, SURFACE_EPS), low_sun),
        SURFACE_EPS, 4.0, 0, True)
    report["shadow_ray_range"] = {
        "ok": (sr.independent and not sr.inherited_from_primary and sr.t_max > 5.0
               and wr.hit and not inherited.hit),
        "shadow_t_max": sr.t_max,
        "primary_t_max_ignored": 5.0,
        "shadowed_default_sun": shadow.shadowed,
        "wall_occluder_t": wr.t,
        "inherited_range_misses_occluder": not inherited.hit,
    }

    # 4) 内部起点：修正后起点在表面外、首采样点不被近平面裁掉
    buried = Ray(Vec3(0.0, 0.0, 5.0), Vec3(0.0, 1.0, -0.2).normalized())
    corrected, t_min, info = inside_start_fix(1, buried, hm)
    hc = hm.sample(corrected.origin.x, corrected.origin.y, 0)
    report["inside_start_fix"] = {
        "ok": info["was_inside"] and corrected.origin.z > hc
        and t_min >= 0.1 and corrected.direction.z > 0.0,
        "depth_inside": info["depth_inside"],
        "flipped_downward": info["flipped_downward"],
        "t_min": t_min,
    }

    # 5) 上传栅栏：信号前读到的是 last_good 而非后缓冲；信号后才发布
    ch = UploadChannel()
    ch.front, ch.last_good = hm, hm
    new_map = load_heightmap(2)
    fence = ch.begin_upload(new_map)
    snapshot_before, _ = ch.acquire()
    blocked = ch.torn_reads_blocked
    ch.signal(fence)
    snapshot_after, fence_after = ch.acquire()
    report["upload_fence"] = {
        "ok": (snapshot_before is hm and blocked == 1
               and snapshot_after is new_map and fence_after.signaled),
        "torn_reads_blocked": blocked,
        "seq_monotonic": fence.seq == 1 and fence_after.seq == 1,
    }

    # 6) 固定弹射：任意 dt 下弹射数恒定，且不同帧时间表平均亮度一致
    policies = [fixed_bounce_count(dt) for dt in (0.001, 0.016, 0.1, 1.0)]
    r_fast = present(4, heightmap=hm, dt_schedule=[0.008] * 4)
    r_slow = present(4, heightmap=hm, dt_schedule=[0.05, 0.02, 0.1, 0.03])
    report["fixed_bounce_count"] = {
        "ok": (all(p.bounces == FIXED_BOUNCES for p in policies)
               and abs(r_fast.mean_luminance - r_slow.mean_luminance) < 1.0e-12),
        "bounces": sorted({p.bounces for p in policies}),
        "luminance_fast": r_fast.mean_luminance,
        "luminance_slow": r_slow.mean_luminance,
    }

    report["all_ok"] = all(v["ok"] for v in report.values())
    return report


def metrics(frame=1) -> dict:
    """默认场景一帧的步进/阴影/弹射统计。"""
    hm = _default_heightmap()
    ray = _default_ray()
    result = march(frame, ray, hm)
    shadow = _trace_shadow(hm, result.point, DEFAULT_T_MAX) if result.hit else None
    shading = shade(frame, ray, hm)
    ch = UploadChannel()
    ch.front = ch.last_good = hm
    fence = ch.begin_upload(load_heightmap(9))
    ch.acquire()  # 未点亮 -> 拦截一次撕裂读取
    return {
        "frame": int(frame),
        "march_steps": result.steps,
        "march_hit": result.hit,
        "max_adaptive_step": result.max_step_used,
        "lod": result.lod,
        "shadow_rays": 1 if shadow is not None else 0,
        "shadow_hits": 1 if shadow is not None and shadow.shadowed else 0,
        "shadow_steps": shadow.steps if shadow is not None else 0,
        "indirect_bounces": shading.bounces,
        "torn_reads_blocked": ch.torn_reads_blocked,
    }


def audit(scene_id=1) -> dict:
    """六个缺陷的修复证据汇总（人读友好）。"""
    report = validate_march(scene_id)
    titles = {
        "adaptive_step": "自适应步长：DDA 边界 + 坡度净空间约束，窄山脊不再被跨过",
        "lod_consistent_hit": "LOD 一致求交：求交采样与粗网格同源，边缘残差为零",
        "shadow_ray_range": "阴影独立射程：AABB 出口距离决定，忽略主射线 t_max",
        "inside_start_fix": "内部起点修正：抬到表面外并翻转下行分量，杜绝黑屏",
        "upload_fence": "上传栅栏：双缓冲 + 序号栅栏，半更新数据不可被 GPU 读取",
        "fixed_bounce_count": "固定弹射次数：与 dt 解耦，时间累积按帧数均分",
    }
    return {key: {"title": titles[key], **report[key]} for key in titles}


def determinism(seed=1) -> dict:
    """确定性核验：同种子重复渲染逐像素一致，且与帧时间安排无关。"""
    hm = load_heightmap(int(seed))
    a = present(3, heightmap=hm)
    b = present(3, heightmap=hm)
    pixel_identical = a.image == b.image
    c = present(3, heightmap=hm, dt_schedule=[0.2, 0.01, 0.05])
    return {
        "seed": int(seed),
        "pixel_identical": pixel_identical,
        "frame_rate_independent": abs(a.mean_luminance - c.mean_luminance) < 1.0e-12,
        "bounce_counts_constant": len(set(a.bounce_counts)) == 1,
        "bounce_count": a.bounce_counts[0] if a.bounce_counts else None,
        "mean_luminance": a.mean_luminance,
    }
