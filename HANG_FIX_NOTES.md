# Hang Fix Summary

## Problem
The process was hanging silently after waveform files were successfully processed. Logs would show "Successfully processed N traces from file X" then stop with no final output.

## Root Causes Identified
1. **Infinite I/O blocking**: ObsPy's `read()` or corrupted MSEED files could hang indefinitely, blocking `as_completed()`
2. **Thread contention**: 20 files/batch with 4 workers caused race conditions on progress_lock during simultaneous completion
3. **No timeout protection**: Futures with no timeout could block indefinitely waiting for I/O-stuck threads
4. **Progress bar deadlock**: `pbar.finish()` might hang when called by multiple threads or after lock contention

## Changes Made to `hypodd_relocator.py`

### 1. Added Timeout Protection (lines 1930-1945)
```python
# Collect results as they complete with timeout protection
# Each file should complete within 300 seconds; if not, mark as failed
for future in as_completed(future_to_file, timeout=300):
    wf_file = future_to_file[future]
    try:
        trace_info_list = future.result(timeout=10)  # Get result with timeout
        ...
    except TimeoutError:
        self.log(f"Timeout reading {wf_file} (likely stuck in I/O), skipping", level="warning")
```
- `as_completed(timeout=300)`: If any future doesn't complete within 5 minutes, raise TimeoutError
- `future.result(timeout=10)`: Retrieve result with 10-second timeout as fallback
- Stuck files are logged and skipped, allowing process to continue

### 2. Reduced Batch Size & Workers (line 1916-1917)
```python
batch_size = 5  # Very small batches for safer processing (was 20)
max_workers = min(2, file_count)  # Even fewer threads (was 4)
```
- Fewer simultaneous threads = less contention on `progress_lock`
- Smaller batches = faster thread completion cycles = lower risk of all threads finishing simultaneously
- Reduces thundering herd effect on progress bar operations

### 3. Added Debug Logging Around pbar.finish() (lines 1966-1971)
```python
self.log(f"All {file_count} waveform files processed, finalizing progress bar...", level="debug")
pbar.finish()
self.log(f"Progress bar finished successfully", level="debug")
```
- Logs before/after `pbar.finish()` to identify where hang occurs
- If hang still happens, logs will indicate whether it's in pbar.finish() or progress bar creation

### 4. Enhanced Error Handling (lines 1936-1945)
- Added explicit handling for `TimeoutError` from stuck I/O operations
- Maintains per-file error tracking with reason ("Timeout (stuck reading)", "Segmentation fault", etc.)
- All failures are logged and tracked, allowing process to continue

## Testing
Created `test_hang_fix.py` to validate:
- Waveform parsing completes without hanging
- Timeouts are triggered appropriately for stuck files
- Progress bar finishes successfully
- Final summary logs are printed

## Expected Behavior After Fix
1. Waveform files process in smaller, safer batches
2. If a file hangs (likely corrupted MSEED), timeout triggers after 300s and file is skipped
3. Progress bar completes without deadlock
4. Final messages print: "Progress bar finished successfully" and performance stats
5. Process exits cleanly with list of any skipped files and reasons

## Further Optimization Options (if hang still occurs)
1. Disable progress bar entirely: `pbar.finish()` → pass (remove progress bar)
2. Use sequential processing for debugging: Set `max_workers = 1` and `batch_size = 1`
3. Add thread dump: Before timeout, print active threads with stack traces
4. Switch to multiprocessing: Avoid GIL/lock contention with separate processes
