import unittest

from detection_engine.realtime_behavior import RealtimeBehaviorDetector


def _make_lm():
    lm = {}
    for i in range(33):
        lm[i] = (0.5, 0.5, 1.0)
    return lm


class RealtimeBehaviorTests(unittest.TestCase):
    def test_concealment_alert_score_in_expected_range(self):
        detector = RealtimeBehaviorDetector(retreat_min=0.05, out_hits_to_reach=1)
        track = "unit-test-track"

        # 第一帧：左手伸直探出，满足 reaching 条件
        lm1 = _make_lm()
        lm1[11] = (0.45, 0.40, 1.0)  # 左肩
        lm1[13] = (0.42, 0.48, 1.0)  # 左肘
        lm1[15] = (0.30, 0.58, 1.0)  # 左手腕（探出）
        lm1[12] = (0.55, 0.40, 1.0)  # 右肩
        lm1[14] = (0.56, 0.50, 1.0)  # 右肘
        lm1[16] = (0.57, 0.58, 1.0)  # 右手腕
        lm1[23] = (0.47, 0.60, 1.0)  # 左髋
        lm1[24] = (0.53, 0.60, 1.0)  # 右髋

        first = detector.step(lm1, track)
        self.assertEqual(first["type"], "Taking Object")

        # 第二帧：左手回缩到胯部藏匿区，触发 Rapid Item Concealment
        lm2 = _make_lm()
        lm2[11] = (0.45, 0.40, 1.0)
        lm2[13] = (0.44, 0.50, 1.0)
        lm2[15] = (0.48, 0.58, 1.0)  # 左手腕回缩到胯部附近
        lm2[12] = (0.55, 0.40, 1.0)
        lm2[14] = (0.56, 0.50, 1.0)
        lm2[16] = (0.57, 0.58, 1.0)
        lm2[23] = (0.47, 0.60, 1.0)
        lm2[24] = (0.53, 0.60, 1.0)

        result = detector.step(lm2, track)

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "Rapid Item Concealment")
        self.assertIn("alert_score", result)
        # P1 校准后：base = 0.50 + 0.30×超额比，下限 0.50（旧公式为 0.45）
        self.assertGreaterEqual(result["alert_score"], 0.50)
        self.assertLessEqual(result["alert_score"], 0.85)
