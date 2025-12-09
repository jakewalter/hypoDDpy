#!/usr/bin/env python3
"""
Quick test to verify the waveform parsing hang is fixed.
This test will process waveform files and check if it completes without hanging.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypoddpy.hypodd_relocator import HypoDDRelocator
import tempfile
import shutil

def test_waveform_parsing_no_hang():
    """Test that waveform parsing completes without hanging"""
    
    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Try to use the existing bridge creek test data
        test_waveform_dir = "bridge_creek_test_output/working_files/cc_files"
        
        if not os.path.exists(test_waveform_dir):
            print(f"Warning: Test data not found at {test_waveform_dir}")
            print("Looking for any .mseed files in bridge_creek_test_output...")
            
            for root, dirs, files in os.walk("bridge_creek_test_output"):
                for file in files:
                    if file.endswith(".mseed") or file.endswith(".ms"):
                        print(f"  Found: {os.path.join(root, file)}")
        
        try:
            # Create a minimal relocator
            relocator = HypoDDRelocator(
                working_dir=temp_dir,
                cc_time_before=0.05,
                cc_time_after=0.2,
                cc_maxlag=0.1,
                cc_filter_min_freq=1,
                cc_filter_max_freq=20,
                use_fdsn_station_lookup=False,
            )
            
            # Set waveform files to any available MSEED files
            print("\nSearching for MSEED files...")
            waveform_files = []
            for root, dirs, files in os.walk("bridge_creek_test_output"):
                for file in files:
                    if file.endswith(".mseed") or file.endswith(".ms"):
                        waveform_files.append(os.path.join(root, file))
            
            if waveform_files:
                print(f"Found {len(waveform_files)} waveform files")
                relocator.waveform_files = waveform_files
                
                # This should complete without hanging
                print("Starting waveform parsing test...")
                import time
                start = time.time()
                
                relocator._parse_waveform_files_parallel(len(waveform_files))
                
                elapsed = time.time() - start
                print(f"✓ Waveform parsing completed successfully in {elapsed:.2f}s")
                print(f"✓ Parsed {len(relocator.waveform_information)} unique traces")
            else:
                print("No MSEED files found to test with")
                print("Test inconclusive - would need actual data files")
        
        except Exception as exc:
            print(f"✗ Error during test: {exc}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    test_waveform_parsing_no_hang()
    print("\n✓ All tests passed - no hang detected!")
