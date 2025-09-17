#!/usr/bin/env python3
"""
Test script to verify trace selection works with fixed channel mapping
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypoddpy.hypodd_relocator import HypoDDRelocator
from obspy import Stream, Trace
import numpy as np

def create_mock_stream(station_id, channels):
    """Create a mock ObsPy stream with specified channels"""
    stream = Stream()

    for channel in channels:
        # Create a simple trace
        data = np.random.randn(1000)
        trace = Trace(data=data)
        trace.stats.station = station_id.split('.')[-1] if '.' in station_id else station_id
        trace.stats.network = station_id.split('.')[0] if '.' in station_id else '*'
        trace.stats.channel = channel
        trace.stats.sampling_rate = 100.0
        stream.append(trace)

    return stream

def test_trace_selection():
    """Test that trace selection works with the fixed channel mapping"""

    # Create a relocator instance
    relocator = HypoDDRelocator(
        working_dir="test_working_dir",
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=1,
        cc_filter_max_freq=20,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.6
    )

    # Test case 1: NM.PENM with HHZ/HHE/HHN channels
    station_id = "NM.PENM"
    channels = ["HHZ", "HHE", "HHN"]
    stream = create_mock_stream(station_id, channels)

    print(f"Testing {station_id} with channels: {channels}")

    # Test selecting Z component
    selected_z = relocator._select_traces_smart(stream, "NM", "PENM", "Z")
    print(f"Selected Z traces: {[tr.stats.channel for tr in selected_z]}")

    # Test selecting E component
    selected_e = relocator._select_traces_smart(stream, "NM", "PENM", "E")
    print(f"Selected E traces: {[tr.stats.channel for tr in selected_e]}")

    # Test selecting N component
    selected_n = relocator._select_traces_smart(stream, "NM", "PENM", "N")
    print(f"Selected N traces: {[tr.stats.channel for tr in selected_n]}")

    # Test case 2: TA.U44A with BHZ/BHE/BHN channels
    station_id2 = "TA.U44A"
    channels2 = ["BHZ", "BHE", "BHN"]
    stream2 = create_mock_stream(station_id2, channels2)

    print(f"\nTesting {station_id2} with channels: {channels2}")

    # Test selecting Z component
    selected_z2 = relocator._select_traces_smart(stream2, "TA", "U44A", "Z")
    print(f"Selected Z traces: {[tr.stats.channel for tr in selected_z2]}")

    # Test selecting E component
    selected_e2 = relocator._select_traces_smart(stream2, "TA", "U44A", "E")
    print(f"Selected E traces: {[tr.stats.channel for tr in selected_e2]}")

    # Test selecting N component
    selected_n2 = relocator._select_traces_smart(stream2, "TA", "U44A", "N")
    print(f"Selected N traces: {[tr.stats.channel for tr in selected_n2]}")

    # Verify selections are correct
    success1 = (len(selected_z) == 1 and selected_z[0].stats.channel == "HHZ" and
                len(selected_e) == 1 and selected_e[0].stats.channel == "HHE" and
                len(selected_n) == 1 and selected_n[0].stats.channel == "HHN")

    success2 = (len(selected_z2) == 1 and selected_z2[0].stats.channel == "BHZ" and
                len(selected_e2) == 1 and selected_e2[0].stats.channel == "BHE" and
                len(selected_n2) == 1 and selected_n2[0].stats.channel == "BHN")

    print(f"\nTest 1 (NM.PENM): {'PASS' if success1 else 'FAIL'}")
    print(f"Test 2 (TA.U44A): {'PASS' if success2 else 'FAIL'}")

    return success1 and success2

if __name__ == "__main__":
    success = test_trace_selection()
    print(f"\nOverall test result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)