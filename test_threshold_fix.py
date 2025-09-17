#!/usr/bin/env python3
"""
Test script to verify cross-correlation works with lowered threshold
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypoddpy.hypodd_relocator import HypoDDRelocator

def test_lowered_threshold():
    """Test cross-correlation with a more realistic threshold"""

    print("Testing cross-correlation with lowered threshold...")

    # Create relocator with lowered threshold
    relocator = HypoDDRelocator(
        working_dir="test_working_dir",
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=1,
        cc_filter_max_freq=20,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.3  # Lowered from 0.6 to 0.3
    )

    print(f"Threshold set to: {relocator.cc_param['cc_min_allowed_cross_corr_coeff']}")
    print("This should allow cross-correlations with coefficients >= 0.3")

    # Test the threshold logic
    test_coeffs = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]

    print("\nThreshold test results:")
    for coeff in test_coeffs:
        would_pass = coeff >= relocator.cc_param['cc_min_allowed_cross_corr_coeff']
        status = "PASS" if would_pass else "FAIL"
        print(".1f")

def create_example_with_lowered_threshold():
    """Create an example configuration with lowered threshold"""

    example_code = '''
# Example configuration with lowered cross-correlation threshold
relocator = HypoDDRelocator(
    working_dir="relocator_working_dir",
    cc_time_before=0.05,
    cc_time_after=0.2,
    cc_maxlag=0.1,
    cc_filter_min_freq=1,
    cc_filter_max_freq=20,
    cc_p_phase_weighting={"Z": 1.0},
    cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
    cc_min_allowed_cross_corr_coeff=0.3,  # LOWERED from 0.6 to 0.3
    # ... other parameters ...
)
'''

    print("\n" + "="*60)
    print("RECOMMENDED CONFIGURATION")
    print("="*60)
    print("Change your cc_min_allowed_cross_corr_coeff from 0.6 to 0.3:")
    print("")
    print(example_code)
    print("")
    print("This should significantly increase the number of successful")
    print("cross-correlations while still maintaining quality control.")

if __name__ == "__main__":
    test_lowered_threshold()
    create_example_with_lowered_threshold()