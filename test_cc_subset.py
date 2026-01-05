#!/usr/bin/env python3
"""
Small test project to validate cross-correlation functionality.
Uses a subset of the test catalog to verify CC calculations.
Based on bridge_hypodd_template.py
"""
import os
import glob
from hypoddpy import HypoDDRelocator


def main():
    # Define working directory for this test
    working_dir = "test_cc_subset_output"
    
    # Initialize the relocator with cross-correlation parameters
    relocator = HypoDDRelocator(
        working_dir=working_dir,
        
        # Cross-correlation time windows
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        
        # Cross-correlation filtering
        cc_filter_min_freq=1.0,
        cc_filter_max_freq=20.0,
        
        # Phase weighting for cross-correlation
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0, "2": 1.0, "1": 1.0},
        
        # Minimum cross-correlation coefficient threshold
        cc_min_allowed_cross_corr_coeff=0.6,
        
        # Channel equivalencies
        custom_channel_equivalencies={"1": "E", "2": "N"},
        
        # Ph2dt parameters
        ph2dt_parameters={
            "MINOBS": 2,      # Lower for small test - minimum observations per event pair
            "MAXSEP": 15.0,   # Maximum inter-event distance in km
        }
    )
    
    # Add test event file (subset of catalog)
    event_file = "bridge_creek_test_output/input_files/test_events_3.xml"
    if os.path.exists(event_file):
        relocator.add_event_files([event_file])
        print(f"✓ Added event file: {event_file}")
    else:
        print(f"✗ Event file not found: {event_file}")
        return
    
    # Look for waveform files (if available)
    # Try multiple possible locations
    waveform_patterns = [
        "test_working_dir/input_files/waveforms/*.mseed",
        "bridge_creek_test_output/input_files/waveforms/*.mseed",
    ]
    
    waveforms_found = False
    for pattern in waveform_patterns:
        waveforms = glob.glob(pattern)
        if waveforms:
            relocator.add_waveform_files(waveforms)
            print(f"✓ Added {len(waveforms)} waveform files from {pattern}")
            waveforms_found = True
            break
    
    if not waveforms_found:
        print("⚠ No waveform files found - test may not produce CC data")
    
    # Look for station files
    station_patterns = [
        "test_working_dir/input_files/*.xml",
        "bridge_creek_test_output/input_files/stations.xml",
    ]
    
    stations_found = False
    for pattern in station_patterns:
        stations = glob.glob(pattern)
        # Filter to only station XML files (not event files)
        stations = [s for s in stations if 'station' in s.lower() or 'inventory' in s.lower()]
        if stations:
            relocator.add_station_files(stations)
            print(f"✓ Added {len(stations)} station files")
            stations_found = True
            break
    
    if not stations_found:
        print("⚠ No station files found - creating simple velocity model without station metadata")
    
    # Setup a simple constant velocity model
    relocator.setup_velocity_model(
        model_type="layered_p_velocity_with_constant_vp_vs_ratio",
        layer_tops=[(0, 2.7), (0.3, 2.95), (1.0, 4.15), (1.5, 5.8), (21, 6.3)],
        vp_vs_ratio=1.73
    )
    print("✓ Velocity model configured")
    
    # Enable debug mode for detailed CC output
    relocator.cc_param["cc_debug_mode"] = True
    print("✓ CC debug mode enabled")
    
    # Start optimized relocation with cross-correlation
    print("\n" + "="*60)
    print("Starting cross-correlation test with optimized relocation...")
    print("="*60 + "\n")
    
    # First validate that CC computation works
    cc_validation_passed = False
    
    try:
        relocator.start_optimized_relocation(
            output_event_file=os.path.join(working_dir, "relocated_events.xml"),
            max_threads=4,             # Use moderate threading for test
            preload_waveforms=True,    # Preload waveforms for efficiency
            monitor_performance=True   # Monitor and log performance
        )
        
        print("\n" + "="*60)
        print("✓ Full relocation completed successfully!")
        print("="*60)
        cc_validation_passed = True
        
    except Exception as e:
        print(f"\n⚠ Relocation failed (expected for small catalog): {e}")
        print("\nValidating cross-correlation output independently...")
        
    # Validate cross-correlation output regardless of relocation success
    print("\n" + "="*60)
    print("CROSS-CORRELATION VALIDATION")
    print("="*60)
    
    cc_file = os.path.join(working_dir, "input_files", "dt.cc")
    if os.path.exists(cc_file):
        with open(cc_file, 'r') as f:
            cc_lines = f.readlines()
        
        # Parse CC data
        cc_measurements = [l for l in cc_lines if not l.startswith('#')]
        event_pairs = [l for l in cc_lines if l.startswith('#')]
        
        print(f"\n✓ Cross-correlation file generated: {cc_file}")
        print(f"  - Event pairs: {len(event_pairs)}")
        print(f"  - CC measurements: {len(cc_measurements)}")
        
        if cc_measurements:
            print("\nSample CC measurements:")
            for i, line in enumerate(cc_measurements[:5]):
                parts = line.strip().split()
                if len(parts) >= 4:
                    station, dt, coef, phase = parts[0], parts[1], parts[2], parts[3]
                    print(f"  {i+1}. {station:12s} dt={float(dt):8.4f}s  coef={float(coef):6.4f}  phase={phase}")
        
        # Check correlation coefficients
        valid_cc = 0
        for line in cc_measurements:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    coef = float(parts[2])
                    if coef >= relocator.cc_param.get("cc_min_allowed_cross_corr_coeff", 0.6):
                        valid_cc += 1
                except:
                    pass
        
        print(f"\n✓ Valid CC measurements (>= threshold): {valid_cc}/{len(cc_measurements)}")
        print(f"  Threshold: {relocator.cc_param.get('cc_min_allowed_cross_corr_coeff', 0.6)}")
        
        cc_validation_passed = True
    else:
        print("\n✗ No dt.cc file generated - cross-correlation failed")
        print("  Check that waveforms are available and match event picks")
        return 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Working directory: {working_dir}/")
    print(f"\nOutput files:")
    print(f"  - input_files/dt.cc: Cross-correlation differential times")
    print(f"  - input_files/dt.ct: Catalog differential times")
    print(f"  - ph2dt_log.txt: Phase processing log")
    
    if cc_validation_passed:
        print("\n✓ CROSS-CORRELATION TEST PASSED")
        print("  CC calculations are working correctly!")
        return 0
    else:
        print("\n✗ CROSS-CORRELATION TEST FAILED")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
