import pytest
import graphics.heightfield_march as mod

CASES = [
    ("adaptive_march_step", (1,)),
    ("lod_consistent_hit", (1,)),
    ("shadow_ray_range", (1,)),
    ("inside_start_fix", (1,)),
    ("upload_fence", (1,)),
    ("fixed_bounce_count", (1,)),
    ("load_heightmap", (1,)),
    ("march", (1,)),
    ("shade", (1,)),
    ("present", (1,)),
    ("validate_march", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
