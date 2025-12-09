from hypoddpy.hypodd_relocator import HypoDDRelocator


def test_fetch_station_metadata_from_fdsn_monkeypatch(monkeypatch, tmp_path):
    # Create relocator with FDSN lookups enabled
    relocator = HypoDDRelocator(
        working_dir=str(tmp_path),
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_filter_min_freq=1.0,
        cc_filter_max_freq=20.0,
        cc_p_phase_weighting={"Z": 1.0},
        cc_s_phase_weighting={"Z": 1.0},
        cc_min_allowed_cross_corr_coeff=0.6,
        use_fdsn_station_lookup=True,
    )

    # Create fake inventory returned by Client.get_stations
    class FakeStation:
        code = 'TEST'
        latitude = 12.34
        longitude = 56.78
        elevation = 123

    class FakeNetwork(list):
        code = 'ZZ'
        def __init__(self):
            super().__init__([FakeStation()])

    class FakeClient:
        def get_stations(self, network=None, station=None, level=None):
            return [FakeNetwork()]

    # Monkeypatch the Client class to return our FakeClient
    monkeypatch.setattr('obspy.clients.fdsn.Client', lambda provider: FakeClient())

    # Call the helper
    result = relocator._fetch_station_metadata_from_fdsn('ZZ', 'TEST')

    assert result is not None
    assert 'ZZ.TEST' in result
    assert 'TEST' in result
    assert result['ZZ.TEST']['latitude'] == 12.34
    assert result['ZZ.TEST']['longitude'] == 56.78
    assert result['ZZ.TEST']['elevation'] == 123
