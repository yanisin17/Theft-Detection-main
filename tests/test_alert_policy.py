import unittest


class AlertPolicyTests(unittest.TestCase):
    def test_low_and_medium_risk_events_do_not_trigger_theft_alert(self):
        from detection_engine.alert_policy import should_trigger_theft_alert

        self.assertFalse(should_trigger_theft_alert('looking_around', 1.0))
        self.assertFalse(should_trigger_theft_alert('suspicious_crouching', 0.99))

    def test_confirmed_concealment_sequence_triggers_theft_alert(self):
        from detection_engine.alert_policy import should_trigger_theft_alert

        # 默认阈值已校准至 0.6：低于 0.6 不报，达标即报
        self.assertFalse(should_trigger_theft_alert('Rapid Item Concealment', 0.59))
        self.assertTrue(
            should_trigger_theft_alert(
                'Rapid Item Concealment', 0.62, sequence_confirmed=True
            )
        )

    def test_confirmed_realtime_event_suppresses_legacy_detector_on_same_frame(self):
        from detection_engine.alert_policy import should_run_aggregate_detector

        self.assertFalse(should_run_aggregate_detector(realtime_confirmed=True))
        self.assertTrue(should_run_aggregate_detector(realtime_confirmed=False))

    def test_other_high_risk_labels_do_not_bypass_sequence_requirement(self):
        from detection_engine.alert_policy import should_trigger_theft_alert

        self.assertFalse(should_trigger_theft_alert('suspicious_item_handling', 0.9))
        self.assertFalse(should_trigger_theft_alert('item_concealment', 0.9))

    def test_alert_requires_sequence_confirmation(self):
        from detection_engine.alert_policy import should_trigger_theft_alert

        self.assertFalse(should_trigger_theft_alert('Rapid Item Concealment', 0.9))
        self.assertTrue(
            should_trigger_theft_alert(
                'Rapid Item Concealment', 0.9, sequence_confirmed=True
            )
        )

    def test_threshold_override_param(self):
        from detection_engine.alert_policy import should_trigger_theft_alert

        # 网格搜索用 threshold 参数覆盖,不影响模块默认值
        self.assertTrue(
            should_trigger_theft_alert(
                'Rapid Item Concealment', 0.65, sequence_confirmed=True, threshold=0.6
            )
        )
        self.assertFalse(
            should_trigger_theft_alert(
                'Rapid Item Concealment', 0.65, sequence_confirmed=True, threshold=0.7
            )
        )

    def test_aggregate_sequence_confirmed_path_can_alert(self):
        """回归测试:聚合路径此前漏传 sequence_confirmed 导致永远无法报警。"""
        from detection_engine.alert_policy import should_trigger_theft_alert

        # backend 聚合路径: sequence_detected=True 且行为归一为 rapid_item_concealment
        agg_behavior = 'rapid_item_concealment'  # sequence_detected=True 时的归一行为
        self.assertTrue(
            should_trigger_theft_alert(
                agg_behavior, 0.75, sequence_confirmed=True
            )
        )
        # 序列未确认时,即使概率很高也不能报警
        self.assertFalse(
            should_trigger_theft_alert(
                'rapid_item_concealment', 0.95, sequence_confirmed=False
            )
        )

    def test_is_alertable_behavior(self):
        from detection_engine.alert_policy import is_alertable_behavior

        self.assertTrue(is_alertable_behavior('Rapid Item Concealment'))
        self.assertTrue(is_alertable_behavior('rapid_item_concealment'))
        self.assertFalse(is_alertable_behavior('item_concealment'))
        self.assertFalse(is_alertable_behavior(None))
        self.assertFalse(is_alertable_behavior(''))


class DetectorConfigTests(unittest.TestCase):
    def test_default_thresholds_match_documented_values(self):
        from detection_engine import detector_config

        # P1 校准后的默认值：阈值 0.6、关闭二次佐证（COCO 测不到货架商品，
        # "手中持物"佐证路径失效，保留会漏报单次真实偷窃）
        self.assertEqual(detector_config.THEFT_ALERT_THRESHOLD, 0.6)
        self.assertEqual(detector_config.RETREAT_MIN, 0.14)
        self.assertEqual(detector_config.OUT_HITS_TO_REACH, 2)
        self.assertFalse(detector_config.REQUIRE_CORROBORATION)
        self.assertEqual(detector_config.CORROBORATION_WINDOW, 10.0)
        # P1：姿态模型升级为 complexity=1（子码流下 0 的关键点抖动过大）
        self.assertEqual(detector_config.MEDIAPIPE_MODEL_COMPLEXITY, 1)

    def test_apply_overrides_updates_known_keys_and_reports_unknown(self):
        import os
        import tempfile
        from detection_engine import detector_config

        with tempfile.NamedTemporaryFile(
            'w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            f.write('{"RETREAT_MIN": 0.2, "NOT_A_REAL_KEY": 1}')
            path = f.name
        try:
            unknown = detector_config.apply_overrides(path)
            self.assertEqual(unknown, ['NOT_A_REAL_KEY'])
            self.assertEqual(detector_config.RETREAT_MIN, 0.2)
        finally:
            os.remove(path)
            # 恢复默认值，避免影响其它测试
            detector_config.RETREAT_MIN = 0.14


if __name__ == '__main__':
    unittest.main()
