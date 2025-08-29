import glob
from hypoddpy import HypoDDRelocator


# Init the relocator with the working directory and some necessary
# configuration values.
relocator = HypoDDRelocator(
    working_dir="relocator_working_dir",
    cc_time_before=0.05,  # Original value
    cc_time_after=0.2,    # Original value
    cc_maxlag=0.1,        # Original value
    cc_filter_min_freq=1, # Original value
    cc_filter_max_freq=20, # Original value
    cc_p_phase_weighting={"Z": 1.0},
    cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
    cc_min_allowed_cross_corr_coeff=0.6, # Optimized: increased from 0.4 (higher quality)
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
    max_threads=8,  # Use more threads for parallel processing
    preload_waveforms=True,  # Preload common waveforms
    monitor_performance=True  # Monitor performance and log speedup
)

# Alternative: Use the standard method with default parameters
# relocator.start_relocation(output_event_file="relocated_events.xml")
