import os
import tempfile
import sys
import runpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from hypoddpy.hypodd_relocator import HypoDDRelocator
from obspy import Stream


def test_parse_with_subprocess_metadata(monkeypatch, tmp_path):
    """If subprocess returns metadata, parsing should create placeholder traces."""
    relocator = HypoDDRelocator(
        working_dir=str(tmp_path),
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=1,
        cc_filter_max_freq=20,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.6,
        use_subprocess_safe_reads=True,
    )

    # Add a fake waveform file name
    fake_file = str(tmp_path / "fake1.mseed")
    # Create an empty placeholder file so validate check passes file existence
    open(fake_file, "wb").close()

    # Make sure validation passes (our test file is empty), and monkeypatch subprocess validator to return metadata
    monkeypatch.setattr(relocator, "_validate_mseed_file", lambda fn: True)
    def fake_read_mseed_in_subprocess(filename, starttime=None, endtime=None, timeout=30):
        return [
            {"id": "XX.FAKE.00.HHZ", "starttime": "2020-01-01T00:00:00.000000Z", "endtime": "2020-01-01T00:01:00.000000Z", "npts": 60, "sampling_rate": 1.0},
        ]

    monkeypatch.setattr(relocator, "_read_mseed_in_subprocess", fake_read_mseed_in_subprocess)

    relocator.waveform_files = [fake_file]
    relocator.waveform_information = {}

    # Call parser (should not hang and should populate waveform_information)
    relocator._parse_waveform_files_parallel(1)

    assert len(relocator.waveform_information) == 1


def test_parse_with_subprocess_timeout(monkeypatch, tmp_path):
    """If subprocess read fails (timeout/None), parsing should skip file and continue."""
    relocator = HypoDDRelocator(
        working_dir=str(tmp_path),
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=1,
        cc_filter_max_freq=20,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0, "E": 1.0, "N": 1.0},
        cc_min_allowed_cross_corr_coeff=0.6,
        use_subprocess_safe_reads=True,
    )

    fake_file = str(tmp_path / "fake2.mseed")
    open(fake_file, "wb").close()

    # Make sure validation passes and simulate subprocess read failing (returns None)
    monkeypatch.setattr(relocator, "_validate_mseed_file", lambda fn: True)
    def failing_read(filename, starttime=None, endtime=None, timeout=30):
        return None

    monkeypatch.setattr(relocator, "_read_mseed_in_subprocess", failing_read)

    relocator.waveform_files = [fake_file]
    relocator.waveform_information = {}

    # This should complete quickly and not hang
    relocator._parse_waveform_files_parallel(1)

    # No traces should be added
    assert len(relocator.waveform_information) == 0
