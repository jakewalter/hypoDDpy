#!/usr/bin/env python3
"""
Quick test script to validate FDSN station lookup functionality.
Tests both the helper method and integration with event reading.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fdsn_lookup_helper():
    """Test the _fetch_station_metadata_from_fdsn helper method."""
    print("=" * 60)
    print("TEST 1: FDSN Station Lookup Helper")
    print("=" * 60)
    
    try:
        from hypoddpy.hypodd_relocator import HypoDDRelocator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create relocator with FDSN lookup enabled
            relocator = HypoDDRelocator(
                working_dir=tmpdir,
                cc_time_before=0.05,
                cc_time_after=0.2,
                cc_maxlag=0.1,
                cc_filter_min_freq=1.0,
                cc_filter_max_freq=20.0,
                cc_p_phase_weighting={"Z": 1.0},
                cc_s_phase_weighting={"Z": 1.0},
                cc_min_allowed_cross_corr_coeff=0.6,
                use_fdsn_station_lookup=True,
            )
            
            print(f"✓ Relocator created with use_fdsn_station_lookup=True")
            print(f"✓ FDSN cache initialized: {hasattr(relocator, '_fdsn_station_cache')}")
            
            # Test fetching a well-known station (IU.ANMO from IRIS)
            print("\nAttempting to fetch station IU.ANMO from IRIS...")
            result = relocator._fetch_station_metadata_from_fdsn('IU', 'ANMO')
            
            if result:
                print(f"✓ FDSN lookup succeeded!")
                print(f"  Stations returned: {list(result.keys())}")
                for station_id, metadata in result.items():
                    print(f"  {station_id}:")
                    print(f"    Latitude: {metadata['latitude']}")
                    print(f"    Longitude: {metadata['longitude']}")
                    print(f"    Elevation: {metadata['elevation']} m")
                
                # Test caching
                print("\nTesting cache...")
                cache_key = ('IU', 'ANMO')
                if cache_key in relocator._fdsn_station_cache:
                    print(f"✓ Cache working: {cache_key} found in cache")
                else:
                    print(f"✗ Cache issue: {cache_key} not found in cache")
                
                return True
            else:
                print(f"✗ FDSN lookup returned None (network issue or station not found)")
                return False
                
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  ObsPy may not be installed or not in PATH")
        return False
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fdsn_integration_with_events():
    """Test that missing stations trigger FDSN lookups during event reading."""
    print("\n" + "=" * 60)
    print("TEST 2: FDSN Integration with Event Reading")
    print("=" * 60)
    
    try:
        from hypoddpy.hypodd_relocator import HypoDDRelocator
        from obspy.core.event import Catalog, Event, Origin, Pick, WaveformStreamID
        from obspy import UTCDateTime
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal QuakeML file with a pick for a station not in station list
            catalog = Catalog()
            event = Event()
            origin = Origin(
                time=UTCDateTime("2024-01-01T00:00:00"),
                latitude=35.0,
                longitude=-106.0,
                depth=5000.0
            )
            event.origins.append(origin)
            event.preferred_origin_id = origin.resource_id
            
            # Create pick for IU.ANMO (a real IRIS station)
            pick = Pick(
                time=UTCDateTime("2024-01-01T00:00:05"),
                phase_hint="P",
                waveform_id=WaveformStreamID(
                    network_code="IU",
                    station_code="ANMO",
                    channel_code="BHZ"
                )
            )
            event.picks.append(pick)
            catalog.events.append(event)
            
            # Write catalog to file
            event_file = os.path.join(tmpdir, "test_event.xml")
            catalog.write(event_file, format="QUAKEML")
            print(f"✓ Created test event file: {event_file}")
            
            # Create relocator WITH FDSN lookup enabled
            relocator_with_fdsn = HypoDDRelocator(
                working_dir=os.path.join(tmpdir, "with_fdsn"),
                cc_time_before=0.05,
                cc_time_after=0.2,
                cc_maxlag=0.1,
                cc_filter_min_freq=1.0,
                cc_filter_max_freq=20.0,
                cc_p_phase_weighting={"Z": 1.0},
                cc_s_phase_weighting={"Z": 1.0},
                cc_min_allowed_cross_corr_coeff=0.6,
                use_fdsn_station_lookup=True,
            )
            
            # Add event file (no station files)
            relocator_with_fdsn.add_event_files([event_file])
            
            print("\nReading events WITH FDSN lookup enabled...")
            relocator_with_fdsn._read_event_information()
            
            # Check if station was added
            if "IU.ANMO" in relocator_with_fdsn.stations or "ANMO" in relocator_with_fdsn.stations:
                print(f"✓ Station was fetched from FDSN and added!")
                print(f"  Stations in relocator: {list(relocator_with_fdsn.stations.keys())}")
                print(f"  Events parsed: {len(relocator_with_fdsn.events)}")
                if relocator_with_fdsn.events:
                    print(f"  Picks in first event: {len(relocator_with_fdsn.events[0]['picks'])}")
                return True
            else:
                print(f"✗ Station was not added (FDSN lookup may have failed)")
                print(f"  Stations in relocator: {list(relocator_with_fdsn.stations.keys())}")
                return False
                
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  ObsPy may not be installed")
        return False
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FDSN Station Lookup Functionality Tests")
    print("=" * 60)
    print("\nNote: These tests require:")
    print("  1. ObsPy to be installed")
    print("  2. Internet connection to reach IRIS FDSN service")
    print("  3. IRIS service to be available")
    print("\n")
    
    results = []
    
    # Test 1: Direct helper method
    results.append(("FDSN Helper Method", test_fdsn_lookup_helper()))
    
    # Test 2: Integration with event reading
    results.append(("FDSN Integration", test_fdsn_integration_with_events()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
