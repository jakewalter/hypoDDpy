#!/usr/bin/env python3
"""
Performance-optimized version of bridge_hypodd_template.py
Includes timing and performance monitoring for waveform identification and CC.
"""
import glob
import time
from hypoddpy import HypoDDRelocator


def main():
    print("="*70)
    print("BRIDGE CREEK RELOCATION - PERFORMANCE OPTIMIZED")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize with optimized parameters
    print("\n1. Initializing HypoDDRelocator with optimized settings...")
    init_start = time.time()
    
    relocator = HypoDDRelocator(
        working_dir="relocate_eqcorr3_optimized",
        
        # Cross-correlation time windows
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        
        # Filtering parameters
        cc_filter_min_freq=1.0,
        cc_filter_max_freq=20.0,
        
        # Phase weighting
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0, "2": 1.0, "1": 1.0},
        
        # Correlation coefficient threshold
        cc_min_allowed_cross_corr_coeff=0.6,
        
        # Channel equivalencies
        custom_channel_equivalencies={"1": "E", "2": "N"},
        
        # Ph2dt parameters - CRITICAL for reducing event pairs
        ph2dt_parameters={
            "MINOBS": 8,      # Increase from 3 - requires more shared observations
            "MAXSEP": 5.0,    # Reduce from 10.0 - only correlate very close events
            "MINLNK": 8,      # Minimum links to form a cluster
        }
    )
    print(f"   ✓ Initialization completed in {time.time() - init_start:.2f}s")
    
    # Add event files
    print("\n2. Adding event files...")
    event_start = time.time()
    event_files = glob.glob("/scratch/bridge_creek/test_location.xml")
    relocator.add_event_files(event_files)
    print(f"   ✓ Added {len(event_files)} event file(s) in {time.time() - event_start:.2f}s")
    
    # Add waveform files
    print("\n3. Scanning for waveform files...")
    waveform_start = time.time()
    waveform_files = glob.glob("/data/okla/ContWaveform/2*/*mseed")
    print(f"   Found {len(waveform_files)} waveform files")
    print(f"   Scan time: {time.time() - waveform_start:.2f}s")
    
    print("\n4. Adding waveform files to relocator...")
    add_wave_start = time.time()
    relocator.add_waveform_files(waveform_files)
    print(f"   ✓ Waveform files added in {time.time() - add_wave_start:.2f}s")
    
    # Enable smart filtering for better performance
    print("\n   Enabling smart waveform filtering...")
    filter_start = time.time()
    relocator.enable_smart_waveform_filtering(buffer_hours=1.0)
    print(f"   ✓ Smart filtering completed in {time.time() - filter_start:.2f}s")
    
    # CRITICAL: Disable excessive debug logging
    relocator.cc_param["cc_debug_mode"] = False
    
    # Add station files
    print("\n5. Adding station files...")
    station_start = time.time()
    station_files = glob.glob("/data/okla/ContWaveform/2*/*.xml")
    relocator.add_station_files(station_files)
    print(f"   ✓ Added {len(station_files)} station file(s) in {time.time() - station_start:.2f}s")
    
    # Setup velocity model
    print("\n6. Setting up velocity model...")
    model_start = time.time()
    relocator.setup_velocity_model(
        model_type="layered_p_velocity_with_constant_vp_vs_ratio",
        layer_tops=[(0, 2.7), (0.3, 2.95), (1.0, 4.15), (1.5, 5.8), (21, 6.3)],
        vp_vs_ratio=1.73
    )
    print(f"   ✓ Velocity model configured in {time.time() - model_start:.2f}s")
    
    # Enable debug mode for detailed output
    relocator.cc_param["cc_debug_mode"] = True
    
    print("\n" + "="*70)
    print("STARTING OPTIMIZED RELOCATION")
    print("="*70)
    print("\nOptimizations enabled:")
    print("  • Parallel processing with 8 threads")
    print("  • Waveform preloading")
    print("  • Performance monitoring")
    print("  • Optimized cross-correlation caching")
    print("")
    
    # Start optimized relocation
    relocation_start = time.time()
    
    try:
        relocator.start_optimized_relocation(
            output_event_file="relocated_events.xml",
            max_threads=8,              # Use 8 threads for parallel processing
            preload_waveforms=True,     # Preload common waveforms
            monitor_performance=True    # Monitor performance and log speedup
        )
        
        relocation_time = time.time() - relocation_start
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("RELOCATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nTiming Summary:")
        print(f"  Initialization:      {init_start:.2f}s")
        print(f"  Event loading:       {event_start:.2f}s")
        print(f"  Waveform discovery:  {waveform_start:.2f}s")
        print(f"  Station loading:     {station_start:.2f}s")
        print(f"  Relocation process:  {relocation_time:.2f}s")
        print(f"  TOTAL TIME:          {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        print(f"\nOutput: relocate_eqcorr3_optimized/relocated_events.xml")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Relocation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
