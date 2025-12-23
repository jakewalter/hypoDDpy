import os
import sys
import json
# Ensure repository root is on sys.path when running tests directly
sys.path.insert(0, os.getcwd())
from hypoddpy.hypodd_relocator import HypoDDRelocator


def test_write_cc_debug_report(tmp_path):
    relocator = HypoDDRelocator(
        working_dir='.',
        cc_time_before=2.0,
        cc_time_after=2.0,
        cc_maxlag=1.0,
        cc_filter_min_freq=0.5,
        cc_filter_max_freq=10.0,
        cc_p_phase_weighting={},
        cc_s_phase_weighting={},
        cc_min_allowed_cross_corr_coeff=0.5,
    )

    # Load existing test files
    base = os.path.join(os.getcwd(), 'bridge_creek_test_output', 'working_files')
    wf_file = os.path.join(base, 'waveform_information.json')
    ev_file = os.path.join(base, 'events.json')
    with open(wf_file) as f:
        relocator.waveform_information = json.load(f)
    with open(ev_file) as f:
        relocator.events = json.load(f)

    out = relocator.write_cc_debug_report(output_path=str(tmp_path / 'cc_debug_report.json'))
    assert os.path.exists(out)
    with open(out) as f:
        rep = json.load(f)
    assert 'failed_files' in rep
    assert 'station_coverage' in rep
    # Station coverage should include at least one station referenced by the events
    assert isinstance(rep['station_coverage'], dict)
    assert len(rep['station_coverage']) > 0
