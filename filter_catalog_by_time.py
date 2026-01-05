#!/usr/bin/env python3
"""
Filter the Bridge Creek catalog to only include events within the waveform time range.
This prevents trying to correlate millions of events with no waveform data.
"""
from obspy import read_events, UTCDateTime

# Read full catalog
print("Reading full catalog...")
catalog = read_events("/scratch/bridge_creek/test_location.xml")
print(f"Total events: {len(catalog)}")

# Define time range where we have waveforms (adjust as needed)
# Based on your waveforms being from 20250201 to 20250222
min_time = UTCDateTime("2025-02-01T00:00:00")
max_time = UTCDateTime("2025-02-23T00:00:00")

# Filter events
filtered = []
for event in catalog:
    if event.preferred_origin():
        origin_time = event.preferred_origin().time
    elif event.origins:
        origin_time = event.origins[0].time
    else:
        continue
    
    if min_time <= origin_time <= max_time:
        filtered.append(event)

filtered_catalog = catalog.__class__(events=filtered)
print(f"Filtered events: {len(filtered_catalog)} ({len(filtered_catalog)/len(catalog)*100:.1f}%)")

# Save filtered catalog
output_file = "/scratch/bridge_creek/test_location_filtered.xml"
filtered_catalog.write(output_file, format="QUAKEML")
print(f"Saved to: {output_file}")

# Calculate expected event pairs
n = len(filtered_catalog)
pairs = n * (n - 1) / 2
print(f"\nExpected event pairs: {pairs:,.0f} (vs original: 5,913,525)")
