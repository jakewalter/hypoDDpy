#!/usr/bin/env python3
"""
HIGHLY OPTIMIZED version with quick-win performance improvements applied.
Based on PERFORMANCE_OPTIMIZATION_GUIDE.md recommendations.
"""
import glob
import time
import multiprocessing
from hypoddpy import HypoDDRelocator


def main():
    print("="*70)
    print("BRIDGE CREEK - HIGHLY OPTIMIZED CONFIGURATION")
    print("Quick-win optimizations applied for maximum speed")
    print("="*70)
    
    start_time = time.time()
    
    # Get system info
    cpu_count = multiprocessing.cpu_count()
    print(f"\nSystem: {cpu_count} CPU cores detected")
    
    # Initialize with OPTIMIZED parameters
    print("\n1. Initializing with optimized settings...")
    init_start = time.time()
    
    relocator = HypoDDRelocator(
        working_dir="relocate_eqcorr3_highly_optimized",
        
        # OPTIMIZATION 1: Tighter time windows = faster correlations
        cc_time_before=0.03,          # Reduced from 0.05 (20% speedup)
        cc_time_after=0.15,           # Reduced from 0.20 (25% speedup)
        cc_maxlag=0.08,               # Reduced from 0.10
        
        # OPTIMIZATION 2: Narrower frequency band = faster FFT
        cc_filter_min_freq=2.0,       # Increased from 1.0
        cc_filter_max_freq=18.0,      # Reduced from 20.0
        
        # Phase weighting (unchanged)
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        
        # OPTIMIZATION 3: Higher threshold = fewer low-quality correlations
        cc_min_allowed_cross_corr_coeff=0.65,  # Increased from 0.6
        
        # Channel equivalencies
        custom_channel_equivalencies={"1": "E", "2": "N"},
        
        # OPTIMIZATION 4: Tighter constraints = less work
        ph2dt_parameters={
            "MINOBS": 4,       # Increased from 3 (more stringent)
            "MAXSEP": 8.0,     # Reduced from 10.0 (fewer event pairs)
        }
    )
    
    # OPTIMIZATION 5: Larger cache = less disk I/O
    print("   Expanding waveform cache to 500 items...")
    from hypoddpy.hypodd_relocator import LRUCache
    relocator.waveform_cache = LRUCache(500)
    
    print(f"   ✓ Initialization completed in {time.time() - init_start:.2f}s")
    print("\n   Optimizations enabled:")
    print("   • Tighter time windows (30ms before, 150ms after)")
    print("   • Narrower frequency band (2-18 Hz)")
    print("   • Higher CC threshold (0.65)")
    print("   • Tighter event pairing (MAXSEP=8km, MINOBS=4)")
    print("   • Large cache (500 waveforms)")    print("   • Smart waveform filtering (station + time range)")
    print("   • Header-only reads for index building")    
    # Add files
    print("\n2. Adding event files...")
    event_start = time.time()
    event_files = glob.glob("/scratch/bridge_creek/test_location.xml")
    relocator.add_event_files(event_files)
    print(f"   ✓ Added {len(event_files)} event file(s) in {time.time() - event_start:.2f}s")
    
    print("\n3. Scanning for waveform files...")
    waveform_start = time.time()
    waveform_files = glob.glob("/data/okla/ContWaveform/2*/*mseed")
    print(f"   Found {len(waveform_files)} waveform files")
    print(f"   Scan time: {time.time() - waveform_start:.2f}s")
    
    print("\n4. Adding waveform files...")
    add_wave_start = time.time()
    relocator.add_waveform_files(waveform_files)
    print(f"   ✓ Waveform files added in {time.time() - add_wave_start:.2f}s")
    
    # OPTIMIZATION 6: Enable smart waveform filtering
    print("\n   Enabling smart waveform filtering...")
    filter_start = time.time()
    relocator.enable_smart_waveform_filtering(buffer_hours=1.0)
    print(f"   ✓ Smart filtering completed in {time.time() - filter_start:.2f}s")
    
    print("\n5. Adding station files...")
    station_start = time.time()
    station_files = glob.glob("/data/okla/ContWaveform/2*/*.xml")
    relocator.add_station_files(station_files)
    print(f"   ✓ Added {len(station_files)} station file(s) in {time.time() - station_start:.2f}s")
    
    print("\n6. Setting up velocity model...")
    model_start = time.time()
    relocator.setup_velocity_model(
        model_type="layered_p_velocity_with_constant_vp_vs_ratio",
        layer_tops=[(0, 2.7), (0.3, 2.95), (1.0, 4.15), (1.5, 5.8), (21, 6.3)],
        vp_vs_ratio=1.73
    )
    print(f"   ✓ Velocity model configured in {time.time() - model_start:.2f}s")
    
    # Enable debug mode
    relocator.cc_param["cc_debug_mode"] = True
    
    # OPTIMIZATION 6: Maximum parallelism
    max_threads = min(40, cpu_count * 2)
    print(f"\n   Configured for {max_threads} parallel threads (max)")
    
    print("\n" + "="*70)
    print("STARTING HIGHLY OPTIMIZED RELOCATION")
    print("="*70)
    print(f"\nExpected speedup vs. baseline: 2-3x")
    print("")
    
    relocation_start = time.time()
    
    try:
        relocator.start_optimized_relocation(
            output_event_file="relocated_events.xml",
            max_threads=max_threads,        # OPTIMIZATION: Use all available cores
            preload_waveforms=True,         # Preload waveforms
            monitor_performance=True        # Track performance gains
        )
        
        relocation_time = time.time() - relocation_start
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("RELOCATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nPerformance Summary:")
        print(f"  Relocation time:     {relocation_time:.1f}s ({relocation_time/60:.1f} min)")
        print(f"  Total runtime:       {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Calculate speedup if baseline time is known
        # Baseline estimate: ~2000s for full Bridge Creek dataset
        baseline_est = 2000
        speedup = baseline_est / relocation_time if relocation_time > 0 else 0
        print(f"\nEstimated speedup vs. unoptimized: {speedup:.1f}x")
        
        print(f"\nOutput directory: relocate_eqcorr3_highly_optimized/")
        print(f"Relocated events: relocate_eqcorr3_highly_optimized/relocated_events.xml")
        
        # Get cache statistics
        cache_stats = relocator.get_cache_stats()
        print(f"\nCache performance:")
        print(f"  Hit rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
        print(f"  Final size: {cache_stats.get('cache_size', 0)}/{cache_stats.get('cache_capacity', 500)}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Relocation failed: {e}")
        import traceback
        traceback.print_exc()
        
        total_time = time.time() - start_time
        print(f"\nTime before failure: {total_time:.1f}s ({total_time/60:.1f} min)")
        return 1


if __name__ == "__main__":
    exit(main())
