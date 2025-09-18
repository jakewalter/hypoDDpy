#!/usr/bin/env python3
"""
Simple test to debug cross-correlation timing issues
"""

from hypoddpy.hypodd_relocator import HypoDDRelocator
import os

def test_cross_correlation_debug():
    # Create relocator
    relocator = HypoDDRelocator(
        working_dir='test_working_dir',
        cc_time_before=0.5,
        cc_time_after=1.0,
        cc_maxlag=0.2,
        cc_filter_min_freq=1.0,
        cc_filter_max_freq=20.0,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.3,
        supress_warning_traces=False
    )

    # Load the data
    relocator.add_waveform_files('test_working_dir/input_files/waveforms/')
    relocator.add_event_files('test_working_dir/input_files/events.xml')
    relocator.add_station_files('test_working_dir/input_files/stations.xml')

    # Get events and picks
    events = relocator.events
    print(f"Found {len(events)} events")

    if len(events) >= 2:
        event_1 = events[0]
        event_2 = events[1]

        picks_1 = relocator._get_picks_for_event(event_1)
        picks_2 = relocator._get_picks_for_event(event_2)

        print(f"Event 1: {event_1['id']}, {len(picks_1)} picks")
        print(f"Event 2: {event_2['id']}, {len(picks_2)} picks")

        # Find common stations
        stations_1 = set(pick['station_id'] for pick in picks_1)
        stations_2 = set(pick['station_id'] for pick in picks_2)
        common_stations = stations_1 & stations_2

        print(f"Common stations: {common_stations}")

        if common_stations:
            station_id = list(common_stations)[0]
            print(f"\nTesting cross-correlation for station: {station_id}")

            # Get picks for this station
            pick_1 = next((p for p in picks_1 if p['station_id'] == station_id), None)
            pick_2 = next((p for p in picks_2 if p['station_id'] == station_id), None)

            if pick_1 and pick_2:
                print(f"Pick 1: time={pick_1['pick_time']}, phase={pick_1['phase']}")
                print(f"Pick 2: time={pick_2['pick_time']}, phase={pick_2['phase']}")

                # Try cross-correlation
                try:
                    time_corr, corr_coeff = relocator._perform_cross_correlation(pick_1, pick_2)
                    print(f"Success: time_correction={time_corr}, correlation={corr_coeff}")
                except Exception as e:
                    print(f"Cross-correlation failed: {e}")

if __name__ == "__main__":
    test_cross_correlation_debug()