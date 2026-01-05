# Performance Optimization Recommendations for hypoDDpy
## Waveform Identification and Cross-Correlation Speed Improvements

Based on analysis of the codebase and current implementation, here are specific recommendations to improve performance:

---

## 1. WAVEFORM IDENTIFICATION OPTIMIZATIONS

### Current Implementation Analysis
The system currently:
- Parses 136,747 waveform files using parallel processing (4-8 threads)
- Uses batch processing with timeout protection
- Caches parsed waveform metadata in JSON
- Has subprocess-safe reads to handle corrupted files

### Recommended Improvements

#### A. **PRE-FILTER Waveform Files (NOW BUILT-IN!)**

**The system now has built-in smart waveform filtering!**

Simply call `enable_smart_waveform_filtering()` after adding waveform files:

```python
# Add all waveform files
relocator.add_waveform_files(glob.glob("/data/waveforms/*.mseed"))

# Enable smart filtering (RECOMMENDED!)
relocator.enable_smart_waveform_filtering(buffer_hours=1.0)

# Now only relevant files will be parsed
relocator.start_relocation("output.xml")
```

**What it does:**
- Analyzes your event catalog to extract time range and station list
- Filters waveforms by parsing filenames (no file I/O needed!)
- Only parses files for stations used in your picks
- Only parses files within event time range (+ buffer)

**Expected speedup:** 5-20x faster waveform parsing!

For Bridge Creek (events spanning a few months of 2 years data): 136k files → ~20k files

**This is now the default in all optimized run scripts!**

#### B. **Use Header-Only Reads for Index Building (IMPLEMENTED!)**

**The code now uses ObsPy's `headonly=True` parameter to read just the metadata:**

```python
# BEFORE (reads full waveform data):
st = self._safe_read_mseed(waveform_file)  # Loads all trace data

# AFTER (reads only headers):
st = read(waveform_file, headonly=True)  # Only reads metadata
```

This extracts trace ID, start time, and end time WITHOUT loading the actual waveform data into memory.

**Expected speedup:** 5-10x faster for building waveform_information.json

**This is now the default behavior!** No configuration needed.

#### C. **Increase Parallel Processing for Waveform Parsing**
```python
# CURRENT: Limited to 8 threads
max_workers = min(8, file_count)

# RECOMMENDED: Scale with CPU count
import multiprocessing
max_workers = min(multiprocessing.cpu_count() * 2, file_count)
# Or explicitly set higher:
max_workers = min(16, file_count)  # For systems with 8+ cores
```

**Expected speedup:** 2-3x faster on multi-core systems

#### D. **Optimize Batch Size for Processing**
```python
# CURRENT: Fixed batch sizes
batch_size = 10  # in some contexts

# RECOMMENDED: Adaptive batch sizing
total_cores = multiprocessing.cpu_count()
batch_size = max(20, total_cores * 5)  # Scale with available resources
```

**Expected speedup:** 15-20% reduction in overhead

#### E. **Skip Already-Parsed Waveforms**
The system already caches waveform_information.json, but you can optimize further:

```python
# Add waveform file change detection
import hashlib
import os

def get_file_signature(filepath):
    """Quick signature: size + mtime"""
    st = os.stat(filepath)
    return f"{st.st_size}_{st.st_mtime}"

# Store signatures with waveform info
# Only reparse if signature changed
```

**Expected speedup:** Near-instant on subsequent runs (currently implemented)

#### F. **Memory-Mapped File Access**
For large continuous waveform files:

```python
# Use memory-mapped I/O for faster access
import mmap

# This is more complex but can reduce I/O time by 30-40%
# Consider for files > 100MB
```

---

## 2. CROSS-CORRELATION OPTIMIZATIONS

### Current Implementation Analysis
- Uses ThreadPoolExecutor with configurable max_threads (default 40)
- LRU cache for waveform data (capacity: 50-200 items)
- Pre-filtering and downsampling with caching
- Smart channel mapping to avoid repeated searches

### Recommended Improvements

#### A. **Increase Cross-Correlation Thread Count**
```python
# CURRENT in template:
max_threads=8

# RECOMMENDED for large datasets:
max_threads=32  # or even 64 on high-core systems

# The code supports up to 40 by default, use it!
relocator._cross_correlate_picks(max_threads=40)
```

**Expected speedup:** 3-5x faster for CC-intensive workloads

#### B. **Increase Waveform Cache Size**
```python
# CURRENT: Auto-adjusted to 50 items for small datasets
self.waveform_cache = LRUCache(50)

# RECOMMENDED for large catalogs with abundant memory:
relocator.waveform_cache = LRUCache(500)  # ~2-5 GB RAM
# or
relocator.waveform_cache = LRUCache(1000)  # ~5-10 GB RAM

# Check your available RAM first:
import psutil
available_gb = psutil.virtual_memory().available / (1024**3)
cache_size = int(available_gb * 50)  # ~50 items per GB
relocator.waveform_cache = LRUCache(cache_size)
```

**Expected speedup:** 40-60% reduction in disk I/O time

#### C. **Enable Aggressive Preloading**
```python
# CURRENT: preload_waveforms=True is good

# RECOMMENDED: Pre-filter event pairs by distance
# Only preload waveforms for stations used by nearby events
ph2dt_parameters={
    "MINOBS": 3,
    "MAXSEP": 10.0,  # Keep this tight to reduce workload
}

# Tighter MAXSEP = fewer event pairs = faster CC
# For initial runs, consider MAXSEP=5.0 or 7.0
```

**Expected speedup:** 2-3x by reducing unnecessary correlations

#### D. **Optimize Filter Parameters**
```python
# CURRENT:
cc_filter_min_freq=1.0
cc_filter_max_freq=20.0

# RECOMMENDED: Consider the frequency content of your data
# Higher min frequency = faster processing
cc_filter_min_freq=2.0   # If signal is strong above 2 Hz
cc_filter_max_freq=15.0  # If no useful signal above 15 Hz

# Narrower filter = faster FFT operations
```

**Expected speedup:** 10-15% faster filtering

#### E. **Use NumPy Acceleration**
Ensure NumPy is built with MKL or OpenBLAS for fastest FFT:

```bash
# Check your NumPy build
python -c "import numpy; numpy.show_config()"

# If not optimized, install:
conda install numpy mkl
# or
pip install numpy[mkl]
```

**Expected speedup:** 20-30% faster FFT operations

---

## 3. MEMORY MANAGEMENT OPTIMIZATIONS

### A. **Garbage Collection Tuning**
```python
import gc

# Before cross-correlation phase
gc.set_threshold(700, 10, 10)  # More aggressive GC
# After intensive operations
gc.collect()
```

### B. **Clear Cache Strategically**
```python
# If processing in phases, clear cache between phases
relocator.clear_waveform_cache()
```

---

## 4. DISK I/O OPTIMIZATIONS

### A. **Use SSD for Working Directory**
```python
# Place working_dir on fastest available storage
relocator = HypoDDRelocator(
    working_dir="/path/to/ssd/relocate_output",  # NOT /scratch or NFS!
    # ...
)
```

**Expected speedup:** 2-10x for I/O-bound operations

### B. **Reduce JSON Serialization Overhead**
```python
# Consider using pickle or msgpack instead of JSON
import pickle

# Replace JSON with pickle for faster serialize/deserialize
with open(cache_file, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

**Expected speedup:** 3-5x faster cache loading

---

## 5. ALGORITHM-LEVEL OPTIMIZATIONS

### A. **Reduce Time Window Sizes**
```python
# CURRENT:
cc_time_before=0.05   # 50 ms before
cc_time_after=0.2     # 200 ms after
cc_maxlag=0.1         # 100 ms max lag

# RECOMMENDED for P-waves (typically sharper):
cc_time_before=0.03   # 30 ms before (sufficient for P)
cc_time_after=0.15    # 150 ms after
cc_maxlag=0.05        # 50 ms max lag (if events are close)

# Smaller windows = less data to correlate = faster
```

**Expected speedup:** 20-40% faster correlations

### B. **Downsample Aggressively**
```python
# If your data is sampled at 100 Hz, downsample to 50 Hz
# The CC code should automatically do this, but verify:
target_sampling_rate = 50  # Hz

# This is controlled internally, but ensure high-freq data is downsampled
```

**Expected speedup:** 2x faster for high-rate data (100+ Hz)

### C. **Skip Poor Quality Correlations Early**
```python
# CURRENT threshold:
cc_min_allowed_cross_corr_coeff=0.6

# RECOMMENDED: Slightly higher to skip marginal correlations
cc_min_allowed_cross_corr_coeff=0.65  # or 0.7

# Fewer low-quality correlations = cleaner results + faster processing
```

---

## 6. MONITORING AND PROFILING

### A. **Enable Detailed Performance Monitoring**
```python
# Already enabled in your test:
monitor_performance=True  # Good!

# Additionally, profile specific sections:
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... run relocation ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(50)  # Top 50 time consumers
```

### B. **Log Cache Hit Rates**
```python
# Add periodic logging
cache_stats = relocator.get_cache_stats()
print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.1%}")
print(f"Cache size: {cache_stats['cache_size']}/{cache_stats['cache_capacity']}")
```

---

## 7. RECOMMENDED CONFIGURATION FOR LARGE CATALOGS

Here's an optimized configuration for the Bridge Creek dataset:

```python
import multiprocessing

relocator = HypoDDRelocator(
    working_dir="relocate_eqcorr3_fast",
    
    # Tighter time windows for speed
    cc_time_before=0.03,          # Reduced from 0.05
    cc_time_after=0.15,           # Reduced from 0.20
    cc_maxlag=0.08,               # Reduced from 0.10
    
    # Optimized filter
    cc_filter_min_freq=2.0,       # Increased from 1.0
    cc_filter_max_freq=18.0,      # Reduced from 20.0
    
    cc_p_phase_weighting={"Z": 1.0},
    cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
    
    # Higher threshold for quality
    cc_min_allowed_cross_corr_coeff=0.65,  # Increased from 0.6
    
    custom_channel_equivalencies={"1": "E", "2": "N"},
    
    # Tighter constraints = less work
    ph2dt_parameters={
        "MINOBS": 4,       # Increased from 3 (more stringent)
        "MAXSEP": 8.0,     # Reduced from 10.0 (focus on closer events)
    }
)

# Larger cache for better performance
relocator.waveform_cache = LRUCache(500)

# Use maximum parallelism
relocator.start_optimized_relocation(
    output_event_file="relocated_events.xml",
    max_threads=min(40, multiprocessing.cpu_count() * 2),  # Scale with hardware
    preload_waveforms=True,
    monitor_performance=True
)
```

---

## 8. EXPECTED OVERALL SPEEDUP

With all optimizations applied:

| Component | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Waveform parsing | ~60s | ~20s | 3x |
| Station parsing | ~10s | ~5s | 2x |
| Cross-correlation | ~1800s | ~450s | 4x |
| **TOTAL** | **~2000s** | **~500s** | **4x** |

*Times are estimates for Bridge Creek dataset scale*

---

## 9. QUICK WINS (Implement These First)

1. **Call `enable_smart_waveform_filtering()`** after adding waveforms (5-20x speedup!) - NOW BUILT-IN!
2. **Header-only reads ALREADY IMPLEMENTED** (5-10x speedup for index building!)
3. **Increase max_threads to 32-40** for cross-correlation
4. **Increase cache size to 500-1000** items
5. **Reduce MAXSEP to 8.0 km** (fewer event pairs)
6. **Increase MINOBS to 4** (higher quality)
7. **Ensure working_dir is on SSD**

**Example usage:**
```python
relocator.add_waveform_files(glob.glob("*.mseed"))
relocator.enable_smart_waveform_filtering()  # <- Add this one line!
relocator.start_optimized_relocation("output.xml", max_threads=40)
```

With #1 and #2 combined, expect **10-50x faster** waveform identification phase!

---

## 10. SYSTEM REQUIREMENTS FOR OPTIMAL PERFORMANCE

- **CPU:** 16+ cores recommended for max_threads=40
- **RAM:** 32GB+ for large cache (500-1000 items)
- **Storage:** SSD for working_dir (NVMe preferred)
- **Network:** Local storage preferred over NFS/network drives

---

## Validation

After implementing optimizations, validate:

1. **Correctness:** Cross-correlation coefficients should be similar
2. **Performance:** Time each phase and compare
3. **Quality:** Check that dt.cc output has similar or better quality

Run your test_cc_subset.py before and after to verify results remain consistent!
