#!/bin/bash
# Clear ph2dt output so it re-runs with new parameters

echo "Clearing ph2dt cached files to force re-run with new parameters..."

rm -f relocate_eqcorr3_optimized/input_files/dt.ct
rm -f relocate_eqcorr3_optimized/input_files/dt.cc  
rm -f relocate_eqcorr3_optimized/input_files/event.sel
rm -f relocate_eqcorr3_optimized/input_files/station.sel
rm -f relocate_eqcorr3_optimized/input_files/event.dat
rm -f relocate_eqcorr3_optimized/input_files/ph2dt.inp
rm -f relocate_eqcorr3_optimized/ph2dt_log.txt

echo "✓ Cleared ph2dt files"
echo ""
echo "Now run: python run_bridge_creek_optimized.py"
echo ""
echo "New parameters:"
echo "  MINOBS: 8 (was 3) - requires more shared observations"
echo "  MAXSEP: 5.0 km (was 10.0) - only nearby events"
echo "  MINLNK: 8 - minimum links for clustering"
echo ""
echo "This should dramatically reduce event pairs!"
