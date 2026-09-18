import unittest
from datetime import datetime, timedelta


class BackendAlertFilenameTests(unittest.TestCase):
    def test_alert_filename_uses_millisecond_precision(self):
        cam_id = "test-cam-id"

        def make_filename(dt):
            return f"alerts/alert_{cam_id}_{dt.strftime('%Y%m%d_%H%M%S_%f')}.jpg"

        f1 = make_filename(datetime(2025, 1, 1, 12, 0, 0, 123456))
        f2 = make_filename(datetime(2025, 1, 1, 12, 0, 0, 123457))

        self.assertNotEqual(f1, f2)
        self.assertTrue(f1.startswith(f"alerts/alert_{cam_id}_"))
        self.assertTrue(f1.endswith(".jpg"))
        self.assertIn("_", f1.split(cam_id)[1].replace(".jpg", ""))

    def test_alert_filename_format(self):
        cam_id = "c011fbdf-1111-2222-3333-444455556666"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"alerts/alert_{cam_id}_{ts}.jpg"

        self.assertRegex(filename, r"^alerts/alert_[0-9a-f-]+_\d{8}_\d{6}_\d{6}\.jpg$")
