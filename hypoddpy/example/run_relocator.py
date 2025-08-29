"""
HypoDD Relocator Example with Advanced Features

This example demonstrates the optimized HypoDD relocator with:
1. Performance optimizations (parallel processing, caching, smart parameters)
2. Smart channel mapping for different seismic data conventions
3. Custom channel equivalencies for networks with specific naming schemes
4. Manual ph2dt parameter control for fine-tuned data selection

NEW FEATURES:
- custom_channel_equivalencies: Map channel names (e.g., {"1": "E", "2": "N"})
- ph2dt_parameters: Override default ph2dt parameters (MINOBS, MAXSEP, etc.)
"""

import glob
from hypoddpy import HypoDDRelocator


# Configuration values with optimized parameters for maximum performance
relocator = HypoDDRelocator(
    working_dir="relocator_working_dir",
    cc_time_before=0.03,  # Optimized: reduced from 0.05 (40% faster)
    cc_time_after=0.12,   # Optimized: reduced from 0.2 (40% faster)
    cc_maxlag=0.05,       # Optimized: reduced from 0.1 (50% faster)
    cc_filter_min_freq=2, # Optimized: increased from 1 (narrower band)
    cc_filter_max_freq=15, # Optimized: reduced from 20 (faster processing)
    cc_p_phase_weighting={"Z": 1.0},  # P-waves: vertical component only
    cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},  # S-waves: all components
    # Note: Smart channel mapping will automatically handle E/N vs 1/2 conversions
    cc_min_allowed_cross_corr_coeff=0.6, # Optimized: increased from 0.4 (higher quality)
    
    # NEW: Custom channel equivalencies for networks where "1"=East, "2"=North
    custom_channel_equivalencies={"1": "E", "2": "N"},
    
    # NEW: Manual ph2dt parameter overrides for fine-tuned control
    ph2dt_parameters={
        "MINOBS": 8,      # Minimum observations per event pair (increased for quality)
        "MAXSEP": 5.0,    # Maximum inter-event distance in km (tightened)
        "MINLNK": 12,     # Minimum links per event (increased)
        "MAXNGH": 4,      # Maximum neighbors (reduced for speed)
    },
    
    # NEW: Disable parallel loading if you encounter corrupted MSEED files
    # disable_parallel_loading=True,  # Uncomment if you get segmentation faults
)

# Add the necessary files. Call a function multiple times if necessary.
relocator.add_event_files(
    "/Users/lion/Documents/Dropbox/Masterarbeit/"
    + "data/final_data/all_obspyck_events.xml"
)
relocator.add_station_files(
    glob.glob(
        "/Users/lion/Documents/Dropbox/"
        + "Masterarbeit/data/final_data/station_data/*.xml"
    )
)
relocator.add_waveform_files(
    glob.glob(
        "/Users/lion/Documents/Dropbox/"
        + "Masterarbeit/data/final_data/waveform_data/*.mseed"
    )
)

# Start the relocation with the desired output file.
# Use the optimized version with optimized parameters for maximum performance
relocator.start_optimized_relocation(
    output_event_file="relocated_events.xml",
    max_threads=8,  # Now properly handled - used for cross-correlation threading
    preload_waveforms=True,  # Preload common waveforms
    monitor_performance=True  # Monitor performance and log speedup
)

# Optional: Get a report of channel mappings that were used
channel_report = relocator.get_channel_mapping_report()
print(f"Channel mapping summary: {channel_report['summary']}")
print(f"Custom equivalencies applied: {relocator.custom_channel_equivalencies}")
print(f"ph2dt parameters used: {relocator.ph2dt_parameters}")

# Alternative: Use the standard method with default parameters
# relocator.start_relocation(output_event_file="relocated_events.xml")
