# Cross-Correlation Subset Test

This test validates that the cross-correlation functionality in hypoDDpy is working correctly using a small subset of the catalog.

## Test Script

**File:** [test_cc_subset.py](test_cc_subset.py)

## What It Tests

- Cross-correlation computation between event pairs
- Waveform processing and filtering
- Phase matching (P and S waves)
- Correlation coefficient calculation
- dt.cc file generation

## Test Results

✓ **PASSED** - Cross-correlation calculations are working correctly

### Generated Data

- **Event pairs:** 3 pairs from 3 events
- **CC measurements:** 4 total measurements
  - 2 P-wave correlations
  - 2 S-wave correlations
- **All measurements exceed threshold:** 0.6 minimum correlation coefficient

### Sample Output

```
Station      dt (s)    Coefficient  Phase
NM.PENM     -0.0007      0.9775       S
TA.U44A     -0.0163      0.8116       P
TA.U44A     -0.0036      0.9009       S
NM.PENM      0.0010      0.9382       P
```

## Configuration

Based on `~/seis/bridge_hypodd_template.py` with parameters:

- **Time windows:** 0.05s before, 0.2s after
- **Max lag:** 0.1s
- **Filter:** 1.0 - 20.0 Hz
- **Min correlation:** 0.6
- **Max separation:** 15 km

## Input Data

- **Events:** 3 events from [bridge_creek_test_output/input_files/test_events_3.xml](bridge_creek_test_output/input_files/test_events_3.xml)
- **Waveforms:** 4 synthetic waveform files
- **Stations:** 2 stations (NM.PENM, TA.U44A)

## Output Location

All results are in: `test_cc_subset_output/`

## Note

The full HypoDD relocation doesn't complete because the catalog is too small to form clusters (minimum 4+ events needed), but this is expected. The test specifically validates that cross-correlation computation is functioning correctly.
