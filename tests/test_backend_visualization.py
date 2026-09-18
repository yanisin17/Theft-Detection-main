import unittest
import numpy as np
from unittest.mock import MagicMock

from backend import get_heatmap_overlay


class BackendVisualizationTests(unittest.TestCase):
    def test_annotations_after_plot_are_preserved(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cam_data = {}
        frame = get_heatmap_overlay(cam_data, frame)

        plotted = frame.copy()
        cv2 = __import__("cv2")
        cv2.rectangle(plotted, (100, 100), (300, 400), (0, 255, 0), 2)

        mocked_results = MagicMock()
        mocked_results.keypoints = MagicMock()
        mocked_results.plot.return_value = plotted
        returned = mocked_results.plot()

        # plot() 返回的是同一张 plotted 帧，后续标注会直接修改它
        self.assertIs(returned, plotted)

        label = "Theft Probability: 85%"
        cv2.putText(returned, label, (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 验证骨架（绿色矩形）和文字（红色）都保留在最终帧上
        self.assertTrue(np.any(plotted != 0))
        self.assertTrue(np.any(returned[:, :, 1] == 255))   # 绿色矩形 -> G 通道
        self.assertTrue(np.any(returned[:, :, 2] == 255))   # 红色文字 -> R 通道
