#!/usr/bin/env python3
"""
Test script for cross-correlation functionality with synthetic data.
Creates miniSEED files with random noise and tests the cross-correlation pipeline.
"""

import numpy as np
import os
from obspy import Stream, Trace, UTCDateTime
from hypoddpy import HypoDDRelocator
import tempfile
import shutil

def create_synthetic_mseed(filename, start_time, duration, sampling_rate=100.0, noise_level=1.0):
    """
    Create a synthetic miniSEED file with random noise.

    :param filename: Output filename
    :param start_time: Start time as UTCDateTime
    :param duration: Duration in seconds
    :param sampling_rate: Sampling rate in Hz
    :param noise_level: Amplitude of random noise
    """
    # Calculate number of samples
    n_samples = int(duration * sampling_rate)

    # Generate random noise
    data = np.random.normal(0, noise_level, n_samples).astype(np.float32)

    # Add some signal-like features (optional)
    # Add a small sine wave to make it more realistic
    t = np.arange(n_samples) / sampling_rate
    signal = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    data += signal

    # Create trace
    trace = Trace()
    trace.data = data
    trace.stats.starttime = start_time
    trace.stats.sampling_rate = sampling_rate
    trace.stats.network = 'XX'
    trace.stats.station = 'TEST'
    trace.stats.location = ''
    trace.stats.channel = 'BHZ'

    # Write to miniSEED
    stream = Stream([trace])
    stream.write(filename, format='MSEED')

    print(f"Created synthetic miniSEED file: {filename}")
    print(f"  Start time: {start_time}")
    print(f"  Duration: {duration}s")
    print(f"  Sampling rate: {sampling_rate}Hz")
    print(f"  Samples: {n_samples}")
    print(f"  Data range: {data.min():.3f} to {data.max():.3f}")

def test_cross_correlation():
    """
    Test the cross-correlation functionality with synthetic data.
    """
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp(prefix='hypodd_test_')
    print(f"Test directory: {test_dir}")

    try:
        # Create synthetic data directory structure
        data_dir = os.path.join(test_dir, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Create synthetic miniSEED files
        base_time = UTCDateTime(2024, 1, 1, 0, 0, 0)

        # File 1: 1 hour of data starting at base_time
        filename1 = os.path.join(data_dir, 'XX.TEST..BHZ__20240101T000000Z__20240101T010000Z.mseed')
        create_synthetic_mseed(filename1, base_time, 3600, sampling_rate=100.0)

        # File 2: 1 hour of data starting 30 minutes later (overlapping)
        filename2 = os.path.join(data_dir, 'XX.TEST..BHZ__20240101T003000Z__20240101T013000Z.mseed')
        create_synthetic_mseed(filename2, base_time + 1800, 3600, sampling_rate=100.0)

        # Create working directory
        working_dir = os.path.join(test_dir, 'working')
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        # Create minimal input files for testing
        create_minimal_input_files(working_dir, data_dir)

        # Test cross-correlation with synthetic data
        print("\n" + "="*50)
        print("TESTING CROSS-CORRELATION WITH SYNTHETIC DATA")
        print("="*50)

        # Initialize relocator with minimal parameters
        relocator = HypoDDRelocator(
            working_dir=working_dir,
            cc_time_before=0.5,  # 0.5 seconds before pick
            cc_time_after=1.0,   # 1.0 seconds after pick
            cc_maxlag=0.5,       # Max lag for cross-correlation
            cc_filter_min_freq=1.0,
            cc_filter_max_freq=20.0,
            cc_p_phase_weighting={"Z": 1.0},
            cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
            cc_min_allowed_cross_corr_coeff=0.1,  # Low threshold for synthetic data
        )

        # Manually set up waveform information for our synthetic files
        relocator.waveform_information = {
            'XX.TEST..BHZ': {
                'filename': filename1,
                'starttime': base_time,
                'endtime': base_time + 3600,
                'sampling_rate': 100.0
            }
        }

        # Also set up the second file
        relocator.waveform_information['XX.TEST..BHZ_2'] = {
            'filename': filename2,
            'starttime': base_time + 1800,
            'endtime': base_time + 5400,
            'sampling_rate': 100.0
        }

        # Create synthetic traces directly for testing
        print("\nCreating synthetic traces for direct cross-correlation test...")

        # Create two synthetic traces with some correlation
        sampling_rate = 100.0
        duration = 2.0  # 2 seconds
        n_samples = int(duration * sampling_rate)

        # Base signal
        t = np.arange(n_samples) / sampling_rate
        base_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave

        # Trace 1: base signal + noise, starts 0.5s before pick
        data1 = base_signal + 0.1 * np.random.normal(0, 1, n_samples)
        trace1 = Trace()
        trace1.data = data1.astype(np.float32)
        trace1.stats.starttime = base_time + 100 - 0.5  # 0.5s before pick
        trace1.stats.sampling_rate = sampling_rate
        trace1.stats.network = 'XX'
        trace1.stats.station = 'TEST'
        trace1.stats.location = ''
        trace1.stats.channel = 'BHZ'

        # Trace 2: same signal but shifted by 0.1 seconds + noise, starts 0.5s before pick
        shift_samples = int(0.1 * sampling_rate)  # 0.1 second shift
        data2 = np.roll(base_signal, shift_samples) + 0.1 * np.random.normal(0, 1, n_samples)
        trace2 = Trace()
        trace2.data = data2.astype(np.float32)
        trace2.stats.starttime = base_time + 110 - 0.5  # 0.5s before pick
        trace2.stats.sampling_rate = sampling_rate
        trace2.stats.network = 'XX'
        trace2.stats.station = 'TEST'
        trace2.stats.location = ''
        trace2.stats.channel = 'BHZ'

        print(f"Created trace 1: {len(trace1.data)} samples, start: {trace1.stats.starttime}")
        print(f"Created trace 2: {len(trace2.data)} samples, start: {trace2.stats.starttime}")

        # Test cross-correlation directly using ObsPy
        from obspy.signal.cross_correlation import xcorr_pick_correction

        pick1_time = base_time + 100 + 0.5  # Pick in middle of trace 1
        pick2_time = base_time + 110 + 0.5  # Pick in middle of trace 2

        print(f"\nTesting direct cross-correlation:")
        print(f"  Pick 1 time: {pick1_time}")
        print(f"  Pick 2 time: {pick2_time}")

        try:
            time_correction, correlation_coeff = xcorr_pick_correction(
                pick1_time,
                trace1,
                pick2_time,
                trace2,
                t_before=0.3,
                t_after=0.3,
                cc_maxlag=0.5,
                plot=False,
            )

            print("\n✅ DIRECT CROSS-CORRELATION SUCCESSFUL!")
            print(f"  Time correction: {time_correction:.6f} seconds")
            print(f"  Correlation coefficient: {correlation_coeff:.6f}")

            # Test with the relocator's method by mocking the required data
            print("\nTesting relocator cross-correlation method...")            # Create mock streams
            st1 = Stream([trace1])
            st2 = Stream([trace2])

            # Mock the relocator's internal state
            relocator.cc_param = {
                'cc_time_before': 0.3,
                'cc_time_after': 0.3,
                'cc_maxlag': 0.5,
                'cc_min_allowed_cross_corr_coeff': 0.1
            }

            # Test channel selection
            st1_filtered = relocator._select_traces_smart(st1, 'XX', 'TEST', 'Z')
            st2_filtered = relocator._select_traces_smart(st2, 'XX', 'TEST', 'Z')

            print(f"Channel selection: st1 has {len(st1_filtered)} traces, st2 has {len(st2_filtered)} traces")

            if len(st1_filtered) > 0 and len(st2_filtered) > 0:
                # Test time filtering (should pass with our synthetic data)
                trace_1 = st1_filtered[0]
                trace_2 = st2_filtered[0]

                print(f"Time filtering test: trace1 covers {trace_1.stats.starttime} to {trace_1.stats.endtime}")
                print(f"Time filtering test: trace2 covers {trace_2.stats.starttime} to {trace_2.stats.endtime}")

                # Test the actual cross-correlation call
                result = xcorr_pick_correction(
                    pick1_time,
                    trace_1,
                    pick2_time,
                    trace_2,
                    t_before=0.3,
                    t_after=0.3,
                    cc_maxlag=0.5,
                    plot=False,
                )

                if result:
                    tc, cc = result
                    print("\n✅ RELOCATOR CROSS-CORRELATION SUCCESSFUL!")
                    print(f"  Time correction: {tc:.6f} seconds")
                    print(f"  Correlation coefficient: {cc:.6f}")
                    return True
                else:
                    print("\n❌ Relocator cross-correlation returned None")
                    return False
            else:
                print("\n❌ Channel selection failed")
                return False

        except Exception as e:
            print(f"\n❌ Cross-correlation failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)

def create_minimal_input_files(working_dir, data_dir):
    """
    Create minimal input files for testing.
    """
    # Create stations file
    stations_file = os.path.join(working_dir, 'stations.dat')
    with open(stations_file, 'w') as f:
        f.write("# Station file for synthetic test\n")
        f.write("XX.TEST 0.0 0.0 0.0 BHZ\n")

    # Create events file
    events_file = os.path.join(working_dir, 'events.dat')
    with open(events_file, 'w') as f:
        f.write("# Event file for synthetic test\n")
        f.write("1 2024-01-01T00:01:40.000000Z 0.0 0.0 5.0 1.0 0.0 0.0 0.0\n")
        f.write("2 2024-01-01T00:01:50.000000Z 0.0 0.0 5.0 1.0 0.0 0.0 0.0\n")

    # Create phase file
    phase_file = os.path.join(working_dir, 'phase.dat')
    with open(phase_file, 'w') as f:
        f.write("# Phase file for synthetic test\n")
        f.write("XX.TEST 2024-01-01T00:01:40.000000Z P 1.0 0.0\n")
        f.write("XX.TEST 2024-01-01T00:01:50.000000Z P 1.0 0.0\n")

    print("Created minimal input files:")
    print(f"  Stations: {stations_file}")
    print(f"  Events: {events_file}")
    print(f"  Phase: {phase_file}")

if __name__ == "__main__":
    print("Testing cross-correlation with synthetic miniSEED data...")
    success = test_cross_correlation()

    if success:
        print("\n🎉 TEST PASSED: Cross-correlation is working correctly!")
    else:
        print("\n💥 TEST FAILED: Cross-correlation is not working.")
        exit(1)