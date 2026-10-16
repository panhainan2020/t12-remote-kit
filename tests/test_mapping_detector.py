import unittest

from t12_remote.mapping_detector import (
    ChannelDetection,
    build_report,
    detect_changed_channel,
)


class TestT12MappingDetector(unittest.TestCase):
    def test_detects_one_changed_channel_with_one_based_label(self):
        detection = detect_changed_channel(
            (1002, 1002, 1002, 282),
            (1002, 1722, 1002, 282),
        )
        self.assertEqual(detection.channel, 2)
        self.assertEqual(detection.baseline_raw, 1002)
        self.assertEqual(detection.active_raw, 1722)
        self.assertEqual(detection.delta, 720)

    def test_rejects_no_change_and_simultaneous_changes(self):
        with self.assertRaisesRegex(ValueError, "没有通道"):
            detect_changed_channel((1002, 1002), (1002, 1050))
        with self.assertRaisesRegex(ValueError, "多个变化通道"):
            detect_changed_channel((1002, 1002), (282, 1722))

    def test_builds_actual_f_g_h_mapping_without_assuming_order(self):
        baseline = (1002,) * 8 + (1722, 282, 282, 1722) + (0,) * 4

        def capture(channel, active):
            center = baseline[channel - 1]
            return ChannelDetection(channel, center, active, active - center)

        captures = {
            "button_e_outer": capture(5, 282),
            "button_e_inner": capture(5, 1722),
            # Exercise a non-positional receiver binding: F=CH8, G=CH6, H=CH7.
            "button_f_outer": capture(8, 282),
            "button_f_inner": capture(8, 1722),
            "button_g_outer": capture(6, 282),
            "button_g_inner": capture(6, 1722),
            "button_h_outer": capture(7, 282),
            "button_h_inner": capture(7, 1722),
            "button_a_pressed": capture(9, 282),
            "button_b_pressed": capture(10, 1722),
            "button_c_pressed": capture(11, 1722),
            "button_d_pressed": capture(12, 282),
        }
        report = build_report(baseline, captures)
        suggestions = report["suggested_calibration"]
        self.assertEqual(suggestions["button_f"]["channel"], 8)
        self.assertEqual(suggestions["button_g"]["channel"], 6)
        self.assertEqual(suggestions["button_h"]["channel"], 7)
        self.assertEqual(suggestions["button_h"]["down"], 282)
        self.assertEqual(suggestions["button_h"]["up"], 1722)
        self.assertEqual(report["errors"], [])

    def test_reports_channel_conflicts(self):
        baseline = (1002,) * 16
        captures = {
            "button_e_outer": ChannelDetection(5, 1002, 282, -720),
            "button_e_inner": ChannelDetection(5, 1002, 1722, 720),
            "button_f_outer": ChannelDetection(5, 1002, 282, -720),
            "button_f_inner": ChannelDetection(5, 1002, 1722, 720),
        }
        report = build_report(baseline, captures)
        self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()

