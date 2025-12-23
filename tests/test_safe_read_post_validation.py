import os
from hypoddpy.hypodd_relocator import HypoDDRelocator


def test_direct_read_after_subprocess_validation(tmp_path):
    # Use a small real mseed file from test_working_dir (exists in repo)
    repo_root = os.path.dirname(os.path.dirname(__file__))
    sample_file = os.path.join(repo_root, 'test_working_dir', 'input_files', 'waveforms', 'NM.PENM.00.HHZ__20120101T120000Z__20120101T120000Z.mseed')
    relocator = HypoDDRelocator(paths={'input_files': '', 'working_files': tmp_path}, use_subprocess_safe_reads=True)

    # Call _safe_read_mseed with a window that should be present.
    # Behavior accepted:
    #  - returns a real Stream with data, or
    #  - raises HypoDDException when the file is unreadable despite subprocess validation.
    try:
        st = relocator._safe_read_mseed(sample_file, None, None)
    except Exception as exc:
        # Accept HypoDDException (or other exceptions indicating unreadable file)
        from hypoddpy.hypodd_relocator import HypoDDException
        assert isinstance(exc, HypoDDException)
        return

    # If we got here, we should have a real Stream with traces
    assert st is not None
    assert len(st) > 0
    has_real_npts = any(getattr(tr.stats, 'npts', 0) > 8 for tr in st)
    assert has_real_npts