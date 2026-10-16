from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from t12_remote.serial import (
    DecodedT12Frame,
    RawSbus25Decoder,
    T12ChannelMapper,
    T12SerialProvider,
    VendorDecoded35Decoder,
    _normalize_channel_config,
    load_t12_config,
)


FIXTURE = Path(__file__).parent / "fixtures" / "t12_test.yaml"


def neutral_channels():
    return (1002,) * 8 + (1722,) * 4 + (0,) * 4


def vendor_frame(channels, status=0):
    words = tuple(channels) + (status,)
    return b"\x0f" + b"".join(int(value).to_bytes(2, "big") for value in words)


class TestT12Decoders(unittest.TestCase):
    def test_vendor_frame_and_fragmented_stream(self):
        channels = neutral_channels()
        frame = vendor_frame(channels)
        decoder = VendorDecoded35Decoder()
        self.assertEqual(len(frame), 35)
        self.assertEqual(decoder.feed(b"garbage" + frame[:9]), [])
        decoded = decoder.feed(frame[9:])
        self.assertEqual(decoded[0].channels, channels)
        self.assertFalse(decoded[0].failsafe)

    def test_vendor_status_masks(self):
        decoder = VendorDecoded35Decoder(lost_mask=1, failsafe_mask=2)
        decoded = decoder.decode(vendor_frame((1002,) * 16, status=3))
        self.assertTrue(decoded.frame_lost)
        self.assertTrue(decoded.failsafe)

    def test_raw_sbus_channels_and_flags(self):
        channels = tuple((index * 97) & 0x7FF for index in range(16))
        packed = sum(value << (11 * index) for index, value in enumerate(channels))
        frame = b"\x0f" + packed.to_bytes(22, "little") + b"\x0c\x00"
        decoded = RawSbus25Decoder().decode(frame)
        self.assertEqual(decoded.channels, channels)
        self.assertTrue(decoded.frame_lost)
        self.assertTrue(decoded.failsafe)

    def test_example_mapping_is_neutral(self):
        config = load_t12_config(FIXTURE)
        mapper = T12ChannelMapper(_normalize_channel_config(config))
        remote = mapper.snapshot(DecodedT12Frame(neutral_channels()), timestamp=12.5)
        np.testing.assert_allclose([remote.x1, remote.y1, remote.x2, remote.y2], 0.0)
        self.assertEqual(
            [remote.button_a, remote.button_b, remote.button_c, remote.button_d],
            [-1, -1, -1, -1],
        )
        self.assertEqual(
            [remote.button_e, remote.button_f, remote.button_g, remote.button_h],
            [0, 0, 0, 0],
        )
        self.assertTrue(remote.connected)

    def test_axis_reversal_is_calibration_only(self):
        config = load_t12_config(FIXTURE)
        channels = deepcopy(_normalize_channel_config(config))
        channels["x1"]["reversed"] = True
        raw = list(neutral_channels())
        raw[0] = 1722
        remote = T12ChannelMapper(channels).snapshot(
            DecodedT12Frame(tuple(raw)), timestamp=1.0
        )
        self.assertAlmostEqual(remote.x1, -1.0)

    def test_provider_marks_stale_snapshot_failsafe(self):
        provider = T12SerialProvider(FIXTURE)
        provider._latest = provider.mapper.snapshot(
            DecodedT12Frame(neutral_channels()), timestamp=1.0
        )
        self.assertTrue(provider.latest(now=1.199).connected)
        stale = provider.latest(now=1.201)
        self.assertFalse(stale.connected)
        self.assertTrue(stale.failsafe)


if __name__ == "__main__":
    unittest.main()

