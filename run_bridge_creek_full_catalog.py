#!/usr/bin/env python3
"""
Full catalog relocation for Bridge Creek using optimized hypoDDpy.
Processes the complete event catalog with smart filtering and parallel processing.
"""
import glob
import time
import argparse
from hypoddpy import HypoDDRelocator


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run full catalog Bridge Creek relocation')
    parser.add_argument('--catalog', type=str, 
                       default='/scratch/bridge_creek/eqcorr3.xml',
                       help='Path to event catalog XML file')
    parser.add_argument('--waveforms', type=str,
                       default='/data/okla/ContWaveform/2*/*mseed',
                       help='Glob pattern for waveform files')
    parser.add_argument('--stations', type=str,
                       default='/data/okla/ContWaveform/2*/*.xml',
                       help='Glob pattern for station XML files')
    parser.add_argument('--output-dir', type=str,
                       default='relocate_full_catalog',
                       help='Working/output directory')
    parser.add_argument('--output-file', type=str,
                       default='relocated_events.xml',
                       help='Output relocated events file name')
    parser.add_argument('--threads', type=int, default=40,
                       help='Number of parallel threads for CC (default: 40)')
    parser.add_argument('--max-sep', type=float, default=10.0,
                       help='Maximum event separation in km (default: 10.0)')
    parser.add_argument('--min-obs', type=int, default=8,
                       help='Minimum observations per event pair (default: 8)')
    parser.add_argument('--min-link', type=int, default=8,
                       help='Minimum links to form cluster (default: 8)')
    parser.add_argument('--cc-threshold', type=float, default=0.6,
                       help='Minimum correlation coefficient (default: 0.6)')
    parser.add_argument('--buffer-hours', type=float, default=2.0,
                       help='Time buffer for smart waveform filtering (default: 2.0)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug output for faster processing')
    
    args = parser.parse_args()
    
    print("="*70)
    print("BRIDGE CREEK FULL CATALOG RELOCATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Catalog:          {args.catalog}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Threads:          {args.threads}")
    print(f"  MAXSEP:           {args.max_sep} km")
    print(f"  MINOBS:           {args.min_obs}")
    print(f"  MINLNK:           {args.min_link}")
    print(f"  CC threshold:     {args.cc_threshold}")
    print(f"  Buffer hours:     {args.buffer_hours}")
    print("")
    
    start_time = time.time()
    
    # Initialize relocator
    print("1. Initializing HypoDDRelocator...")
    init_start = time.time()
    
    relocator = HypoDDRelocator(
        working_dir=args.output_dir,
        
        # Cross-correlation time windows (from original hypoDDpy)
        cc_time_before=0.05,    # 50ms before pick
        cc_time_after=0.2,      # 200ms after pick
        cc_maxlag=0.1,          # 100ms maximum lag
        
        # Bandpass filter for cross-correlation
        cc_filter_min_freq=1.0,
        cc_filter_max_freq=20.0,
        
        # Phase weighting - weight all three components equally
        cc_p_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0, "1": 1.0, "2": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0, "1": 1.0, "2": 1.0},
        
        # Correlation coefficient threshold
        cc_min_allowed_cross_corr_coeff=args.cc_threshold,
        
        # Channel equivalencies for different instrument types
        custom_channel_equivalencies={"1": "E", "2": "N"},
        
        # Ph2dt parameters - controls event pair selection
        ph2dt_parameters={
            "MINOBS": args.min_obs,    # Minimum shared observations
            "MAXSEP": args.max_sep,    # Maximum event separation (km)
            "MINLNK": args.min_link,   # Minimum links for clustering
        }
    )
    print(f"   ✓ Initialization completed in {time.time() - init_start:.2f}s")
    
    # Add event catalog
    print("\n2. Loading event catalog...")
    event_start = time.time()
    event_files = glob.glob(args.catalog)
    if not event_files:
        print(f"   ✗ ERROR: No event files found matching '{args.catalog}'")
        return 1
    
    relocator.add_event_files(event_files)
    print(f"   ✓ Added {len(event_files)} event file(s) in {time.time() - event_start:.2f}s")
    
    # Scan for waveform files
    print("\n3. Scanning for waveform files...")
    waveform_start = time.time()
    waveform_files = glob.glob(args.waveforms)
    print(f"   Found {len(waveform_files):,} waveform files")
    print(f"   Scan time: {time.time() - waveform_start:.2f}s")
    
    if len(waveform_files) == 0:
        print(f"   ✗ ERROR: No waveform files found matching '{args.waveforms}'")
        return 1
    
    # Add waveforms with smart filtering
    print("\n4. Adding waveform files to relocator...")
    add_wave_start = time.time()
    relocator.add_waveform_files(waveform_files)
    print(f"   ✓ Waveform files added in {time.time() - add_wave_start:.2f}s")
    
    print("\n   Enabling smart waveform filtering...")
    filter_start = time.time()
    relocator.enable_smart_waveform_filtering(buffer_hours=args.buffer_hours)
    print(f"   ✓ Smart filtering completed in {time.time() - filter_start:.2f}s")
    
    # Add station files
    print("\n5. Loading station metadata...")
    station_start = time.time()
    station_files = glob.glob(args.stations)
    if len(station_files) == 0:
        print(f"   ✗ ERROR: No station files found matching '{args.stations}'")
        return 1
    
    relocator.add_station_files(station_files)
    print(f"   ✓ Added {len(station_files):,} station file(s) in {time.time() - station_start:.2f}s")
    
    # Setup velocity model for Oklahoma
    print("\n6. Setting up velocity model...")
    model_start = time.time()
    relocator.setup_velocity_model(
        model_type="layered_p_velocity_with_constant_vp_vs_ratio",
        # Oklahoma velocity model
        layer_tops=[
            (0.0, 2.7),    # Surface layer
            (0.3, 2.95),   # Shallow sediments
            (1.0, 4.15),   # Mid sediments
            (1.5, 5.8),    # Basement
            (21.0, 6.3)    # Lower crust
        ],
        vp_vs_ratio=1.73
    )
    print(f"   ✓ Velocity model configured in {time.time() - model_start:.2f}s")
    
    # Configure debug mode
    if args.no_debug:
        relocator.cc_param["cc_debug_mode"] = False
        print("\n   Debug mode: DISABLED (faster processing)")
    else:
        relocator.cc_param["cc_debug_mode"] = True
        print("\n   Debug mode: ENABLED (detailed logging)")
    
    # Start relocation
    print("\n" + "="*70)
    print("STARTING RELOCATION")
    print("="*70)
    print("\nOptimizations:")
    print(f"  • Parallel processing: {args.threads} threads")
    print(f"  • Smart waveform filtering: ±{args.buffer_hours}h buffer")
    print(f"  • Event pair constraints: MAXSEP={args.max_sep}km, MINOBS={args.min_obs}")
    print(f"  • Correlation threshold: ≥{args.cc_threshold}")
    print("")
    
    relocation_start = time.time()
    
    try:
        # Use standard relocation (start_relocation) which now has our optimizations
        relocator.start_relocation(
            output_event_file=args.output_file,
            output_cross_correlation_file=f"{args.output_dir}/cc_results.json",
            create_plots=False  # Disable plots for large catalogs
        )
        
        relocation_time = time.time() - relocation_start
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("RELOCATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nTiming Summary:")
        print(f"  Initialization:      {time.time() - init_start:.2f}s")
        print(f"  Event loading:       {time.time() - event_start:.2f}s")
        print(f"  Waveform discovery:  {time.time() - waveform_start:.2f}s")
        print(f"  Waveform filtering:  {time.time() - filter_start:.2f}s")
        print(f"  Station loading:     {time.time() - station_start:.2f}s")
        print(f"  Relocation process:  {relocation_time:.2f}s ({relocation_time/60:.1f}m)")
        print(f"  TOTAL TIME:          {total_time:.2f}s ({total_time/60:.1f}m, {total_time/3600:.1f}h)")
        
        print(f"\nOutput files:")
        print(f"  Relocated events:    {args.output_dir}/{args.output_file}")
        print(f"  CC results:          {args.output_dir}/cc_results.json")
        print(f"  Working files:       {args.output_dir}/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n✗ Relocation interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n✗ Relocation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
