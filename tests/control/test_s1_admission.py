"""S1 admission respects the explicitly registered R2 reservation release."""
import importlib.util
from pathlib import Path

s=importlib.util.spec_from_file_location('s1_admission',Path('scripts/launch_s1_after_v2_drains.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_only_all_stopped_records_release_s1():
    assert not m.eligible([{'status':'stopped_v2'},{'status':'draining_v2'},{'status':'stopped_v2'}])
    assert m.eligible([{'status':'stopped_v2'}]*3)
