#!/usr/bin/env python3
"""
Create synthetic seismic data and run cross-correlation to generate pick files
"""

import os
import numpy as np
from obspy import UTCDateTime, Stream, Trace
from obspy.core.event import Event, Origin, Pick, WaveformStreamID, ResourceIdentifier
from obspy.core.event import Catalog, Magnitude
import glob

def create_synthetic_events():
    """Create synthetic earthquake events with picks - simplified version"""
    # Events are now created directly in XML format below
    pass

def create_waveforms_for_test():
    """Create synthetic waveforms for the test events"""

    # Create output directory
    waveform_dir = "/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/waveforms"
    os.makedirs(waveform_dir, exist_ok=True)

    # Event times
    event_times = [
        UTCDateTime("2012-01-01T12:00:00.000000Z"),
        UTCDateTime("2012-01-01T12:01:00.000000Z")
    ]

    # Stations
    stations = [("NM", "PENM"), ("TA", "U44A")]

    for event_idx, origin_time in enumerate(event_times):
        for network, station in stations:
            # Create synthetic seismic signal
            duration = 30  # 30 seconds
            sampling_rate = 100.0
            n_samples = int(duration * sampling_rate)

            t = np.linspace(0, duration, n_samples)

            # P-wave and S-wave arrival times (from the XML)
            if station == "PENM":
                p_arrival = 2.5 if event_idx == 0 else 2.48
                s_arrival = 4.5 if event_idx == 0 else 4.52
            else:  # U44A
                p_arrival = 1.8 if event_idx == 0 else 1.75
                s_arrival = 3.8 if event_idx == 0 else 3.85

            # P-wave signal (higher frequency)
            p_signal = np.exp(-((t - p_arrival) ** 2) / (2 * 0.1 ** 2)) * np.sin(2 * np.pi * 10 * t)

            # S-wave signal (lower frequency, larger amplitude)
            s_signal = 2 * np.exp(-((t - s_arrival) ** 2) / (2 * 0.2 ** 2)) * np.sin(2 * np.pi * 5 * t)

            # Background noise
            noise = 0.1 * np.random.randn(n_samples)

            # Combine signals
            signal = p_signal + s_signal + noise

            # Create traces for 3 components
            components = ['HHZ', 'HHE', 'HHN']
            stream = Stream()

            for comp in components:
                trace = Trace(data=signal + 0.05 * np.random.randn(n_samples))
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.location = "00"
                trace.stats.channel = comp
                trace.stats.sampling_rate = sampling_rate
                trace.stats.starttime = origin_time
                stream.append(trace)

            # Save to MSEED file
            filename = f"{network}.{station}.00.{components[0]}__{origin_time.strftime('%Y%m%dT%H%M%SZ')}__{origin_time.strftime('%Y%m%dT%H%M%SZ')}.mseed"
            filepath = os.path.join(waveform_dir, filename)
            stream.write(filepath, format='MSEED')

            print(f"Created waveform file: {filename}")

def create_station_file():
    """Create a simple station XML file"""

    from obspy.core.inventory import Inventory, Network, Station, Channel, Response

    # Create inventory
    inventory = Inventory(networks=[], source="synthetic")

    # Add networks and stations
    for network_code, station_code in [("NM", "PENM"), ("TA", "U44A")]:
        network = Network(code=network_code, stations=[])

        station = Station(code=station_code,
                         latitude=35.0,
                         longitude=-120.0,
                         elevation=500.0)

        # Add channels
        for channel_code in ["HHZ", "HHE", "HHN"]:
            channel = Channel(code=channel_code,
                            location_code="00",
                            latitude=35.0,
                            longitude=-120.0,
                            elevation=500.0,
                            depth=0.0)
            station.channels.append(channel)

        network.stations.append(station)
        inventory.networks.append(network)

    # Save to file
    station_file = "/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/stations.xml"
    inventory.write(station_file, format="STATIONXML")
    print(f"Created station file: {station_file}")

    return station_file

def run_cross_correlation():
    """Run the cross-correlation with lowered threshold"""

    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, '/Users/jwalter/seis/hypoDDpy')
    from hypoddpy import HypoDDRelocator

    print("\n" + "="*60)
    print("RUNNING CROSS-CORRELATION WITH LOWERED THRESHOLD")
    print("="*60)

    # Create relocator with lowered threshold
    relocator = HypoDDRelocator(
        working_dir="/Users/jwalter/seis/hypoDDpy/test_working_dir",
        cc_time_before=0.5,   # Increased for better overlap
        cc_time_after=1.0,    # Increased for better overlap
        cc_maxlag=0.2,        # Increased to allow more lag
        cc_filter_min_freq=1,
        cc_filter_max_freq=20,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.3,  # LOWERED FROM 0.6 TO 0.3
        supress_warning_traces=True  # Reduce log noise
    )

    # Add files
    event_file = "/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/events.xml"
    station_file = "/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/stations.xml"
    waveform_files = glob.glob("/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/waveforms/*.mseed")

    print(f"Event file: {event_file}")
    print(f"Station file: {station_file}")
    print(f"Waveform files: {len(waveform_files)} files")

    relocator.add_event_files(event_file)
    relocator.add_station_files([station_file])
    relocator.add_waveform_files(waveform_files)

    # Run cross-correlation only (don't do full relocation)
    try:
        print("\nStarting cross-correlation...")
        relocator.start_optimized_relocation(
            output_event_file="/Users/jwalter/seis/hypoDDpy/test_working_dir/output_events.xml",
            max_threads=2,  # Use fewer threads for testing
            preload_waveforms=True
        )
        print("Cross-correlation completed successfully!")

        # Check if dt.cc files were created
        dt_files = glob.glob("/Users/jwalter/seis/hypoDDpy/test_working_dir/*.dt.cc*")
        print(f"\nGenerated {len(dt_files)} cross-correlation files:")
        for f in dt_files:
            print(f"  {os.path.basename(f)}")

    except Exception as e:
        print(f"Error during cross-correlation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Creating synthetic seismic data for cross-correlation testing...")

    # Create synthetic data
    create_synthetic_events()

    # Save events to file (simple XML format)
    event_file = "/Users/jwalter/seis/hypoDDpy/test_working_dir/input_files/events.xml"
    os.makedirs(os.path.dirname(event_file), exist_ok=True)

    # Create simple XML content
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<quakeml xmlns="http://quakeml.org/xmlns/quakeml/1.2" xmlns:qml="http://quakeml.org/xmlns/quakeml/1.2">
  <eventParameters publicID="smi:local/EventParameters">
    <event publicID="smi:local/Event/1">
      <origin publicID="smi:local/Origin/1">
        <time><value>2012-01-01T12:00:00.000000Z</value></time>
        <latitude><value>35.0</value></latitude>
        <longitude><value>-120.0</value></longitude>
        <depth><value>5000.0</value></depth>
      </origin>
      <pick publicID="smi:local/Pick/1">
        <time><value>2012-01-01T12:00:02.500000Z</value></time>
        <waveformID networkCode="NM" stationCode="PENM" locationCode="00" channelCode="HHZ"/>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/Pick/2">
        <time><value>2012-01-01T12:00:04.500000Z</value></time>
        <waveformID networkCode="NM" stationCode="PENM" locationCode="00" channelCode="HHZ"/>
        <phaseHint>S</phaseHint>
      </pick>
      <pick publicID="smi:local/Pick/3">
        <time><value>2012-01-01T12:00:01.800000Z</value></time>
        <waveformID networkCode="TA" stationCode="U44A" locationCode="00" channelCode="HHZ"/>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/Pick/4">
        <time><value>2012-01-01T12:00:03.800000Z</value></time>
        <waveformID networkCode="TA" stationCode="U44A" locationCode="00" channelCode="HHZ"/>
        <phaseHint>S</phaseHint>
      </pick>
    </event>
    <event publicID="smi:local/Event/2">
      <origin publicID="smi:local/Origin/2">
        <time><value>2012-01-01T12:01:00.000000Z</value></time>
        <latitude><value>35.01</value></latitude>
        <longitude><value>-120.01</value></longitude>
        <depth><value>5100.0</value></depth>
      </origin>
      <pick publicID="smi:local/Pick/5">
        <time><value>2012-01-01T12:01:02.480000Z</value></time>
        <waveformID networkCode="NM" stationCode="PENM" locationCode="00" channelCode="HHZ"/>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/Pick/6">
        <time><value>2012-01-01T12:01:04.520000Z</value></time>
        <waveformID networkCode="NM" stationCode="PENM" locationCode="00" channelCode="HHZ"/>
        <phaseHint>S</phaseHint>
      </pick>
      <pick publicID="smi:local/Pick/7">
        <time><value>2012-01-01T12:01:01.750000Z</value></time>
        <waveformID networkCode="TA" stationCode="U44A" locationCode="00" channelCode="HHZ"/>
        <phaseHint>P</phaseHint>
      </pick>
      <pick publicID="smi:local/Pick/8">
        <time><value>2012-01-01T12:01:03.850000Z</value></time>
        <waveformID networkCode="TA" stationCode="U44A" locationCode="00" channelCode="HHZ"/>
        <phaseHint>S</phaseHint>
      </pick>
    </event>
  </eventParameters>
</quakeml>'''

    with open(event_file, 'w') as f:
        f.write(xml_content)
    print(f"Created event file: {event_file}")

    # Create station file
    create_station_file()

    # Create waveforms manually for the two events
    create_waveforms_for_test()

    # Run cross-correlation
    run_cross_correlation()