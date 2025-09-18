#!/usr/bin/env python3
"""
Test cross-correlation with optimized parameters for real seismic data
"""

import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypoddpy.hypodd_relocator import HypoDDRelocator

def test_real_data_cross_correlation():
    """Test cross-correlation with real seismic data using optimized parameters"""

    print("Testing cross-correlation with optimized parameters for real data...")
    print("="*70)

    # Create relocator with optimized parameters for real data
    relocator = HypoDDRelocator(
        working_dir="/Users/jwalter/seis/hypoDDpy/test_working_dir",
        cc_time_before=0.5,      # Increased from 0.03/0.05 for better overlap
        cc_time_after=1.0,       # Increased from 0.12/0.2 for better overlap
        cc_maxlag=0.2,           # Increased from 0.05/0.1 to allow more lag
        cc_filter_min_freq=1,    # Reduced from 2 for broader frequency band
        cc_filter_max_freq=20,   # Increased from 15 for broader frequency band
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.3,  # Reduced from 0.6 for real data
        supress_warning_traces=True  # Reduce log noise
    )

    # Add files
    event_file = "/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/events.xml"
    station_file = "/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/stations.xml"
    waveform_files = glob.glob("/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/waveforms/*.mseed")

    print(f"Found {len(waveform_files)} waveform files")
    print(f"Event file: {event_file}")
    print(f"Station file: {station_file}")
    print()

    # Add files to relocator
    relocator.add_event_files([event_file])
    relocator.add_station_files([station_file])
    relocator.add_waveform_files(waveform_files)

    # Setup velocity model (required for relocation)
    relocator.setup_velocity_model(
        model_type="layered_p_velocity_with_constant_vp_vs_ratio",
        vp_vs_ratio=1.73,  # Typical crustal value
        layer_tops=[
            (0.0, 3.5),    # Surface layer: 3.5 km/s
            (2.0, 5.0),    # Upper crust: 5.0 km/s
            (15.0, 6.5),   # Lower crust: 6.5 km/s
            (30.0, 8.0)    # Upper mantle: 8.0 km/s
        ]
    )

    # Configure ph2dt parameters for better event pairing
    relocator.ph2dt_parameters.update({
        "MAXSEP": 10.0,   # Maximum inter-event distance in km (increased from 0)
        "MINOBS": 4,      # Minimum observations per event pair (decreased)
        "MINLNK": 6,      # Minimum links per event (decreased)
    })

    print("Velocity model and ph2dt parameters configured.")
    print(f"ph2dt MAXSEP: {relocator.ph2dt_parameters.get('MAXSEP', 'default')}")
    print()

    print("Files added to relocator. Starting cross-correlation analysis...")
    print("Parameters:")
    print(f"  cc_time_before: {relocator.cc_param['cc_time_before']}s")
    print(f"  cc_time_after: {relocator.cc_param['cc_time_after']}s")
    print(f"  cc_maxlag: {relocator.cc_param['cc_maxlag']}s")
    print(f"  cc_min_allowed_cross_corr_coeff: {relocator.cc_param['cc_min_allowed_cross_corr_coeff']}")
    print()

    try:
        # Start the relocation process (this will run cross-correlation)
        relocator.start_optimized_relocation("test_output.xml", max_threads=4)

        print("\n✅ SUCCESS: Cross-correlation analysis completed!")
        print("Check the log file for detailed results.")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Check the log file for more details.")

if __name__ == "__main__":
    test_real_data_cross_correlation()