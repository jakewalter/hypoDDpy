#!/usr/bin/env python3
"""
OPTIMIZED Bridge Creek relocation with SMART WAVEFORM FILTERING
Only parses waveforms that are actually needed based on:
1. Event time range
2. Stations used in picks
"""
import glob
import time
import multiprocessing
from obspy import read_events, UTCDateTime
from hypoddpy import HypoDDRelocator


def get_event_time_range_and_stations(event_file):
    """Extract time range and station list from event catalog"""
    print(f"  Analyzing event catalog: {event_file}")
    catalog = read_events(event_file)
    
    min_time = None
    max_time = None
    stations_used = set()
    
    for event in catalog:
        # Get origin time
        if event.preferred_origin():
            origin_time = event.preferred_origin().time
        elif event.origins:
            origin_time = event.origins[0].time
        else:
            continue
            
        if min_time is None or origin_time < min_time:
            min_time = origin_time
        if max_time is None or origin_time > max_time:
            max_time = origin_time
        
        # Collect stations from picks
        for pick in event.picks:
            station_code = pick.waveform_id.station_code
            stations_used.add(station_code)
    
    # Add buffer for waveform windows
    buffer_hours = 1.0  # 1 hour buffer on each side
    min_time = min_time - buffer_hours * 3600
    max_time = max_time + buffer_hours * 3600
    
    print(f"  ✓ Event time range: {min_time} to {max_time}")
    print(f"  ✓ Duration: {(max_time - min_time) / 86400:.1f} days")
    print(f"  ✓ Stations used: {len(stations_used)} unique stations")
    print(f"    Sample stations: {list(stations_used)[:10]}")
    
    return min_time, max_time, stations_used


def filter_waveform_files(all_files, min_time, max_time, stations_used):
    """Filter waveform files by time range and station"""
    print(f"\n  Filtering {len(all_files)} waveform files...")
    
    filtered = []
    for wf_file in all_files:
        # Parse filename: typically contains station and time info
        # Format: NETWORK.STATION.LOC.CHAN__STARTTIME__ENDTIME.mseed
        basename = wf_file.split('/')[-1]
        parts = basename.split('__')
        
        if len(parts) >= 2:
            # Extract station from first part (NETWORK.STATION.LOC.CHAN)
            net_sta_parts = parts[0].split('.')
            if len(net_sta_parts) >= 2:
                station = net_sta_parts[1]
                
                # Check if station is in our list
                if station not in stations_used:
                    continue
                
                # Extract time from filename
                try:
                    # Parse start time from filename
                    time_str = parts[1]  # e.g., "20250221T000000Z"
                    file_time = UTCDateTime(time_str)
                    
                    # Check if file time overlaps with our event range
                    # Allow 1 day buffer since files are typically 1 day long
                    if file_time > max_time + 86400:
                        continue
                    if len(parts) >= 3:
                        end_time_str = parts[2].replace('.mseed', '')
                        file_end = UTCDateTime(end_time_str)
                        if file_end < min_time - 86400:
                            continue
                except:
                    # If we can't parse the filename, include it to be safe
                    pass
        
        filtered.append(wf_file)
    
    reduction = (1 - len(filtered) / len(all_files)) * 100
    print(f"  ✓ Filtered to {len(filtered)} files ({reduction:.1f}% reduction)")
    print(f"  ✓ This will skip parsing {len(all_files) - len(filtered)} unnecessary files")
    
    return filtered


def main():
    print("="*70)
    print("BRIDGE CREEK - SMART WAVEFORM FILTERING")
    print("Only parse waveforms that are actually needed!")
    print("="*70)
    
    start_time = time.time()
    
    # STEP 1: Analyze event catalog FIRST
    print("\n1. Analyzing event catalog to determine requirements...")
    event_file = "/scratch/bridge_creek/test_location.xml"
    
    min_time, max_time, stations_used = get_event_time_range_and_stations(event_file)
    
    # STEP 2: Filter waveform files BEFORE adding to relocator
    print("\n2. Filtering waveform files based on event requirements...")
    all_waveform_files = glob.glob("/data/okla/ContWaveform/2*/*mseed")
    print(f"  Total available: {len(all_waveform_files)} files")
    
    filtered_waveform_files = filter_waveform_files(
        all_waveform_files, min_time, max_time, stations_used
    )
    
    # STEP 3: Initialize relocator
    print("\n3. Initializing HypoDDRelocator...")
    cpu_count = multiprocessing.cpu_count()
    
    relocator = HypoDDRelocator(
        working_dir="relocate_eqcorr3_smart_filtered",
        cc_time_before=0.03,
        cc_time_after=0.15,
        cc_maxlag=0.08,
        cc_filter_min_freq=2.0,
        cc_filter_max_freq=18.0,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.65,
        custom_channel_equivalencies={"1": "E", "2": "N"},
        ph2dt_parameters={
            "MINOBS": 4,
            "MAXSEP": 8.0,
        }
    )
    
    # Increase cache
    from hypoddpy.hypodd_relocator import LRUCache
    relocator.waveform_cache = LRUCache(500)
    print("  ✓ Initialized with optimized settings")
    
    # STEP 4: Add filtered files
    print("\n4. Adding event files...")
    relocator.add_event_files([event_file])
    
    print("\n5. Adding FILTERED waveform files...")
    add_start = time.time()
    relocator.add_waveform_files(filtered_waveform_files)
    print(f"  ✓ Added in {time.time() - add_start:.2f}s")
    
    print("\n6. Adding station files...")
    station_files = glob.glob("/data/okla/ContWaveform/2*/*.xml")
    relocator.add_station_files(station_files)
    
    print("\n7. Setting up velocity model...")
    relocator.setup_velocity_model(
        model_type="layered_p_velocity_with_constant_vp_vs_ratio",
        layer_tops=[(0, 2.7), (0.3, 2.95), (1.0, 4.15), (1.5, 5.8), (21, 6.3)],
        vp_vs_ratio=1.73
    )
    
    relocator.cc_param["cc_debug_mode"] = False  # Disable verbose debug
    
    max_threads = min(40, cpu_count * 2)
    
    print("\n" + "="*70)
    print("STARTING RELOCATION WITH SMART FILTERING")
    print("="*70)
    
    speedup_est = len(all_waveform_files) / len(filtered_waveform_files)
    print(f"\nExpected waveform parsing speedup: {speedup_est:.1f}x")
    print("")
    
    relocation_start = time.time()
    
    try:
        relocator.start_optimized_relocation(
            output_event_file="relocated_events.xml",
            max_threads=max_threads,
            preload_waveforms=True,
            monitor_performance=True
        )
        
        relocation_time = time.time() - relocation_start
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Output: relocate_eqcorr3_smart_filtered/relocated_events.xml")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
