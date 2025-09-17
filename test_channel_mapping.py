#!/usr/bin/env python3
"""
Test script to verify channel mapping fix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypoddpy.hypodd_relocator import HypoDDRelocator

def test_channel_mapping():
    """Test the smart channel mapping functionality"""

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
    available_channels = ["NM.PENM.00.HHZ", "NM.PENM.00.HHE", "NM.PENM.00.HHN"]

    print(f"Testing {station_id} with channels: {available_channels}")

    # Extract just the channel codes (not full IDs)
    channel_codes = [ch.split('.')[-1] for ch in available_channels]
    print(f"Channel codes: {channel_codes}")

    mapping = relocator._get_smart_channel_mapping(station_id, channel_codes)
    print(f"Channel mapping: {mapping}")

    # Test case 2: TA.U44A with BHZ/BHE/BHN channels
    station_id2 = "TA.U44A"
    available_channels2 = ["TA.U44A..BHZ", "TA.U44A..BHE", "TA.U44A..BHN"]

    print(f"\nTesting {station_id2} with channels: {available_channels2}")

    channel_codes2 = [ch.split('.')[-1] for ch in available_channels2]
    print(f"Channel codes: {channel_codes2}")

    mapping2 = relocator._get_smart_channel_mapping(station_id2, channel_codes2)
    print(f"Channel mapping: {mapping2}")

    # Verify mappings are correct (core Z/E/N mappings)
    expected_mapping1 = {'Z': 'HHZ', 'E': 'HHE', 'N': 'HHN'}
    expected_mapping2 = {'Z': 'BHZ', 'E': 'BHE', 'N': 'BHN'}

    # Check that core mappings are present
    success1 = all(mapping.get(k) == v for k, v in expected_mapping1.items())
    success2 = all(mapping2.get(k) == v for k, v in expected_mapping2.items())

    print(f"\nTest 1 (NM.PENM): {'PASS' if success1 else 'FAIL'}")
    print(f"Expected: {expected_mapping1}")
    print(f"Got: {mapping}")

    print(f"\nTest 2 (TA.U44A): {'PASS' if success2 else 'FAIL'}")
    print(f"Expected: {expected_mapping2}")
    print(f"Got: {mapping2}")

    return success1 and success2

if __name__ == "__main__":
    success = test_channel_mapping()
    print(f"\nOverall test result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)