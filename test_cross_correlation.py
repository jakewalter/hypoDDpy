#!/usr/bin/env python3
"""
Test script to check cross-correlation coefficients and debug why correlations are failing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypoddpy.hypodd_relocator import HypoDDRelocator
from obspy import Stream, Trace
from obspy.signal.cross_correlation import xcorr_pick_correction
import numpy as np

def create_mock_waveforms():
    """Create mock waveforms with some correlation"""
    # Create two similar waveforms with some noise
    np.random.seed(42)  # For reproducible results

    # Base signal
    t = np.linspace(0, 1, 1000)
    base_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

    # Add some noise
    noise1 = 0.1 * np.random.randn(1000)
    noise2 = 0.1 * np.random.randn(1000)

    # Create traces
    trace1 = Trace(data=base_signal + noise1)
    trace1.stats.sampling_rate = 100.0
    trace1.stats.station = 'TEST'
    trace1.stats.channel = 'HHZ'

    trace2 = Trace(data=base_signal + noise2)
    trace2.stats.sampling_rate = 100.0
    trace2.stats.station = 'TEST'
    trace2.stats.channel = 'HHZ'

    return trace1, trace2

def test_realistic_cross_correlation():
    """Test cross-correlation with realistic parameters matching the relocator"""

    print("\nTesting with realistic parameters...")

    # Create waveforms with more realistic characteristics
    np.random.seed(42)

    # Create longer traces (10 seconds at 100 Hz = 1000 samples)
    n_samples = 1000
    sampling_rate = 100.0

    # Create a seismic-like signal with P and S phases
    t = np.linspace(0, 10, n_samples)

    # P-wave signal (higher frequency, smaller amplitude)
    p_signal = 0.5 * np.exp(-((t - 2.0) ** 2) / (2 * 0.1 ** 2)) * np.sin(2 * np.pi * 15 * t)

    # S-wave signal (lower frequency, larger amplitude)
    s_signal = 2.0 * np.exp(-((t - 4.0) ** 2) / (2 * 0.2 ** 2)) * np.sin(2 * np.pi * 8 * t)

    # Background noise
    noise_level = 0.1
    noise1 = noise_level * np.random.randn(n_samples)
    noise2 = noise_level * np.random.randn(n_samples)

    # Create traces
    trace1 = Trace(data=p_signal + s_signal + noise1)
    trace1.stats.sampling_rate = sampling_rate
    trace1.stats.station = 'TEST'
    trace1.stats.channel = 'HHZ'

    trace2 = Trace(data=p_signal + s_signal + noise2)
    trace2.stats.sampling_rate = sampling_rate
    trace2.stats.station = 'TEST'
    trace2.stats.channel = 'HHZ'

    # Test P-wave correlation (pick at 2.0 seconds)
    pick_time1 = 2.0
    pick_time2 = 2.0

    print("Testing P-wave correlation (similar signals):")
    try:
        corr_time, corr_coeff = xcorr_pick_correction(
            pick_time1, trace1, pick_time2, trace2,
            t_before=0.05, t_after=0.2, cc_maxlag=0.1, plot=False
        )
        print(".3f")
        print(f"  Time correction: {corr_time:.6f} seconds")
    except Exception as e:
        print(f"  Error: {e}")

    # Test with slightly different pick times (simulating pick uncertainty)
    print("\nTesting P-wave correlation (with pick uncertainty):")
    pick_time2_uncertain = 2.02  # 20ms uncertainty

    try:
        corr_time, corr_coeff = xcorr_pick_correction(
            pick_time1, trace1, pick_time2_uncertain, trace2,
            t_before=0.05, t_after=0.2, cc_maxlag=0.1, plot=False
        )
        print(".3f")
        print(f"  Time correction: {corr_time:.6f} seconds")
    except Exception as e:
        print(f"  Error: {e}")

    # Test S-wave correlation
    print("\nTesting S-wave correlation:")
    s_pick_time1 = 4.0
    s_pick_time2 = 4.0

    try:
        corr_time, corr_coeff = xcorr_pick_correction(
            s_pick_time1, trace1, s_pick_time2, trace2,
            t_before=0.05, t_after=0.2, cc_maxlag=0.1, plot=False
        )
        print(".3f")
        print(f"  Time correction: {corr_time:.6f} seconds")
    except Exception as e:
        print(f"  Error: {e}")

def test_threshold_analysis():
    """Analyze what correlation coefficients are needed for successful cross-correlation"""

    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

    print("Current threshold: 0.6 (60%)")
    print("This means cross-correlations must have >60% similarity")
    print("")
    print("Typical correlation coefficients in seismology:")
    print("- Perfect match: 1.0")
    print("- Very good match: 0.8-0.9")
    print("- Good match: 0.6-0.8")
    print("- Moderate match: 0.4-0.6")
    print("- Poor match: 0.2-0.4")
    print("- No correlation: ~0.0")
    print("")
    print("If your data has pick uncertainties >20-30ms,")
    print("correlation coefficients will naturally be lower.")
    print("")
    print("RECOMMENDATIONS:")
    print("1. Try threshold = 0.3-0.4 for initial testing")
    print("2. Check pick quality and timing accuracy")
    print("3. Verify waveforms have sufficient signal-to-noise ratio")
    print("4. Consider increasing cc_time_before/cc_time_after if signals are weak")

def test_relocator_threshold():
    """Test the relocator with different correlation thresholds"""

    print("\nTesting relocator with different thresholds...")

    # Test with very low threshold (should always pass)
    relocator_low = HypoDDRelocator(
        working_dir="test_working_dir",
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=1,
        cc_filter_max_freq=20,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.0  # Very low threshold
    )

    print(f"Low threshold relocator cc_min_allowed_cross_corr_coeff: {relocator_low.cc_param['cc_min_allowed_cross_corr_coeff']}")

    # Test with current threshold
    relocator_normal = HypoDDRelocator(
        working_dir="test_working_dir",
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=1,
        cc_filter_max_freq=20,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.6  # Current threshold
    )

    print(f"Normal threshold relocator cc_min_allowed_cross_corr_coeff: {relocator_normal.cc_param['cc_min_allowed_cross_corr_coeff']}")

if __name__ == "__main__":
    test_realistic_cross_correlation()
    test_relocator_threshold()
    test_threshold_analysis()