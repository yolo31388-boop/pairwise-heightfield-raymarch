"""六个缺陷修复的行为级测试。

与只检查“返回非 None”的 ``test_heightfield_march.py`` 不同，本文件用对抗性
场景验证每条修复真正消除了题目描述的故障现象：

1. 固定大步长跨过窄山脊 -> 自适应步长必须命中；
2. 全分辨率求交与粗 LOD 网格错开 -> 命中点必须落在该 LOD 表面上；
3. 阴影沿用主射线射程 -> 独立射程能判到山后遮挡，沿用射程会漏判；
4. 相机在地形内部、射线朝下 -> 修正后起点在表面外、首采样点越过近平面；
5. 高度图异步上传半更新 -> 栅栏点亮前 GPU 只能读到上一完整快照；
6. 间接光弹射数随帧率变化 -> 任意 dt 下弹射数恒定、平均亮度逐位一致。
"""

import math

import pytest

import graphics.heightfield_march as mod


# --- 修复 1：自适应步长 ---------------------------------------------------


def test_adaptive_step_hits_narrow_ridge_that_fixed_step_skips():
    ridge = mod._ridge_heightmap()
    ray = mod.Ray(mod.Vec3(-30.0, 0.0, 10.0), mod.Vec3(1.0, 0.0, 0.0).normalized())

    fixed = mod._march_core(ridge, ray, 0.0, 200.0, 0, adaptive=False, fixed_step=12.0)
    adaptive = mod._march_core(ridge, ray, 0.0, 200.0, 0, adaptive=True)

    assert fixed.hit is False          # 复现“整段跨过山脊打到背后天空”
    assert adaptive.hit is True
    # 命中点必须贴近墙所在的 x=0 纵面，而不是墙后
    assert abs(adaptive.point.x) < ridge.lod_cell(0)
    assert adaptive.steps > 0


def test_adaptive_step_is_bounded_and_positive():
    step = mod.adaptive_march_step(1)
    assert step > 0.0
    step_fine = mod.adaptive_march_step(
        1.0, clearance=0.01, dir_xy=0.99, dir_z=-0.1, slope=2.0)
    step_loose = mod.adaptive_march_step(
        1.0, clearance=100.0, dir_xy=0.1, dir_z=-0.99, slope=0.1)
    # 净空间越小 / 坡面越陡，步长越保守
    assert 0.0 < step_fine < step_loose


# --- 修复 2：LOD 一致求交 -------------------------------------------------


@pytest.mark.parametrize("lod", [0, 1, 2, 3, 4])
def test_lod_consistent_hit_lies_on_same_lod_surface(lod):
    hm = mod._default_heightmap()
    result = mod.lod_consistent_hit(lod, heightmap=hm)
    assert result.hit
    surface_z = hm.sample(result.point.x, result.point.y, lod)
    # 命中点与该 LOD 粗网格表面残差为零（旧实现这里会有错位，形成锯齿洞）
    assert abs(result.point.z - surface_z) < 1e-6


def test_naive_fullres_surface_differs_from_coarse_lod():
    # 证明“全分辨率表面”与“粗 LOD 表面”确实不同，因此必须按 LOD 求交
    hm = mod._default_heightmap()
    hit = mod.lod_consistent_hit(3, heightmap=hm)
    fullres_z = hm.sample(hit.point.x, hit.point.y, 0)
    lod_z = hm.sample(hit.point.x, hit.point.y, 3)
    assert abs(fullres_z - lod_z) > 1e-4


# --- 修复 3：阴影独立射程 -------------------------------------------------


def test_shadow_range_independent_of_primary_t_max():
    hm = mod._default_heightmap()
    point = mod.Vec3(10.0, 10.0, hm.sample(10.0, 10.0, 0))
    sr = mod.shadow_ray_range(
        1, point, heightmap=hm, primary_t_max=3.0, primary_remaining=0.5)
    assert sr.independent is True
    assert sr.inherited_from_primary is False
    assert sr.t_max > 3.0          # 不被主射线的短射程限制


def test_shadow_detects_occluder_only_with_independent_range():
    wall = mod._ridge_heightmap()
    low_sun = mod.Vec3(-1.0, 0.0, 0.25).normalized()
    point = mod.Vec3(20.0, 0.0, 0.0)
    sr = mod.shadow_ray_range(
        1, point, sun=low_sun, heightmap=wall, primary_t_max=4.0)
    ray = mod.Ray(point + mod.Vec3(0.0, 0.0, mod.SURFACE_EPS), low_sun)

    independent = mod._march_core(wall, ray, sr.t_min, sr.t_max, 0, True)
    inherited = mod._march_core(wall, ray, mod.SURFACE_EPS, 4.0, 0, True)

    assert independent.hit is True       # 独立射程：墙后正确落在阴影里
    assert inherited.hit is False        # 沿用主射线射程：墙在射程外，漏判


# --- 修复 4：内部起点修正 -------------------------------------------------


def test_inside_start_lifts_origin_above_surface_and_flips_down_ray():
    hm = mod._default_heightmap()
    buried = mod.Ray(mod.Vec3(0.0, 0.0, 5.0),
                     mod.Vec3(0.0, 1.0, -0.2).normalized())
    corrected, t_min, info = mod.inside_start_fix(1, buried, hm, near_plane=0.1)

    assert info["was_inside"] is True
    assert info["depth_inside"] > 0.0
    surface = hm.sample(corrected.origin.x, corrected.origin.y, 0)
    assert corrected.origin.z > surface
    assert corrected.direction.z > 0.0      # 朝下的分量被翻转朝外
    assert info["flipped_downward"] is True
    assert t_min >= 0.1                     # 首采样点不被近平面裁掉

    # 修正后的射线能够正常求交，不会再整片发黑
    result = mod._march_core(hm, corrected, t_min, mod.DEFAULT_T_MAX, 0, True)
    assert result.reason in ("surface_crossing", "left_heightmap_domain",
                             "max_distance_reached")


def test_inside_start_is_noop_when_above_surface():
    hm = mod._default_heightmap()
    above = mod.Ray(mod.Vec3(0.0, 0.0, 150.0),
                    mod.Vec3(0.0, 1.0, -1.0).normalized())
    corrected, t_min, info = mod.inside_start_fix(1, above, hm)
    assert info["was_inside"] is False
    assert corrected.origin == above.origin


# --- 修复 5：上传栅栏 -----------------------------------------------------


def test_upload_fence_blocks_half_updated_read_until_signaled():
    hm = mod._default_heightmap()
    channel = mod.UploadChannel()
    channel.front = channel.last_good = hm
    new_map = mod.load_heightmap(7)

    fence = channel.begin_upload(new_map)
    assert fence.signaled is False

    snapshot_before, fence_before = channel.acquire()
    # 栅栏未点亮：GPU 只能继续读上一完整快照，绝不能读到后缓冲的半更新数据
    assert snapshot_before is hm
    assert snapshot_before is not channel.back
    assert fence_before.signaled is False
    assert channel.torn_reads_blocked == 1

    channel.signal(fence)
    snapshot_after, fence_after = channel.acquire()
    assert fence_after.signaled is True
    assert snapshot_after is new_map
    assert channel.back is None              # 发布后后缓冲已交换


def test_fallback_returns_last_good_snapshot_not_torn_back_buffer():
    hm = mod._default_heightmap()
    channel = mod.UploadChannel()
    channel.front = channel.last_good = hm
    channel.begin_upload(mod.load_heightmap(11))
    snapshot, _ = channel.acquire()
    assert snapshot is channel.last_good
    assert snapshot is not channel.back


def test_fence_sequence_monotonic():
    channel = mod.UploadChannel()
    channel.front = channel.last_good = mod._default_heightmap()
    f1 = channel.begin_upload(mod.load_heightmap(1))
    f2 = channel.begin_upload(mod.load_heightmap(2))
    assert f2.seq > f1.seq


# --- 修复 6：固定弹射次数 -------------------------------------------------


@pytest.mark.parametrize("dt", [0.001, 0.016, 0.1, 1.0, 5.0])
def test_bounce_count_constant_regardless_of_frame_time(dt):
    policy = mod.fixed_bounce_count(dt)
    assert policy.bounces == mod.FIXED_BOUNCES
    assert policy.frame_rate_independent is True


def test_luminance_identical_across_frame_rate_schedules():
    hm = mod._default_heightmap()
    fast = mod.present(4, heightmap=hm, dt_schedule=[0.008] * 4)
    slow = mod.present(4, heightmap=hm,
                       dt_schedule=[0.05, 0.02, 0.1, 0.03])
    # 弹射数恒定 + 时间累积按帧数均分 -> 不同机器/帧率下亮度逐位一致
    assert set(fast.bounce_counts) == {mod.FIXED_BOUNCES}
    assert fast.image == slow.image
    assert fast.mean_luminance == slow.mean_luminance


# --- 整体：validate / determinism / 管线 ---------------------------------


def test_validate_report_all_ok():
    report = mod.validate_march()
    assert report.pop("all_ok") is True
    for name, evidence in report.items():
        assert evidence["ok"] is True, name


def test_determinism_pixel_identical_and_framerate_independent():
    d = mod.determinism(1)
    assert d["pixel_identical"] is True
    assert d["frame_rate_independent"] is True
    assert d["bounce_counts_constant"] is True
    assert d["bounce_count"] == mod.FIXED_BOUNCES


def test_default_ray_hits_terrain_at_all_lods():
    hm = mod._default_heightmap()
    ray = mod._default_ray()
    for lod in range(5):
        result = mod.march(1, ray, hm, lod=lod)
        assert result.hit is True
        assert result.reason == "surface_crossing"
