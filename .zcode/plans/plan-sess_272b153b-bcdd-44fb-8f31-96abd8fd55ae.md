# 提升监控识别精度方案

## 现状结论(探索结果)

- 报警链路:YOLOv8n-pose 追踪 → MediaPipe 33 点实时状态机(伸手→回缩藏匿)→ `alert_policy.py` 判定(≥0.7 报警);旧 21 行为聚合引擎只做画面标注。
- **关键缺陷**:`backend.py:1269` 聚合路径调用 `should_trigger_theft_alert` 未传 `sequence_confirmed=True`,该函数逻辑上永远返回 False → 聚合引擎的序列检测(抓取→藏匿组合)即使命中也无法报警,系统实际只有一条报警通路,漏报风险高。
- **无评估体系**:无 ground truth、无 precision/recall 脚本。可用资产:SQLite 31 条报警记录 + `alerts/` 截图 + `evidence_videos/` 17 段各 300 帧证据视频 + `tmp_eval_frames/` 3 帧。

## 阶段 1:建立评估体系(先量化基线)

1. **`tools/export_alerts_for_labeling.py`**:从 SQLite 导出全部报警记录 → 生成 `labels/alerts_to_label.csv`(alert_id、截图路径、证据视频路径、时间、置信度、行为类型),留 `is_true_positive` 空列由你人工填 1/0。
2. **`tools/eval_detection.py`** 离线评估器,两种模式:
   - **精度模式**:读标注 CSV → 输出 precision、误报按行为/场景分类统计。
   - **查全模式**:指定包含已知偷窃的视频目录 + 时间区间标注 → 离线回放检测流水线(复用 `RealtimeBehaviorDetector` + `alert_policy`,读视频文件而非摄像头)→ 输出 recall、报警延迟、漏报明细。
   - 注意:查全评估需要你提供"包含偷窃但未必触发过报警"的视频(证据视频都是已报警片段,只能算精度)。
3. 跑出**基线报告**(当前参数下的 P/R/F1)作为后续所有优化的对照。

## 阶段 2:修复 + 规则调优(快速见效)

4. **修复聚合报警死路**:`backend.py:1269` 传入 `sequence_confirmed=p_state.sequence_detected`;序列确认时把可报警行为归一为 `rapid_item_concealment`。同步更新 `tests/test_alert_policy.py`。
5. **实时路径降误报(佐证确认机制)**:单次藏匿事件先记 pending,满足任一条件才真报警:(a) 同一 track 短时间窗(默认 10s)内出现第二次藏匿事件;(b) `check_object_in_hand` 在藏匿后确认手中持物。带开关可配置,避免过度收紧造成漏报。
6. **阈值集中化**:把 `realtime_behavior.py` 的 `_RETREAT_MIN`、`_OUT_HITS_TO_REACH`、`_REACH_TIMEOUT`、`THEFT_ALERT_THRESHOLD`、MediaPipe `model_complexity`(现为 0,升 1 可提姿态精度)统一收进配置,便于评估调优。
7. **网格搜索调参**:用评估集对关键阈值(_RETREAT_MIN、THEFT_ALERT_THRESHOLD、确认窗口)做小网格搜索,选 F1 最优组合,输出调优前后对比报告。

## 阶段 3:模型升级路径(中期)

8. **`tools/train_shoplifting.py`**:`backend.py:900` 本就优先加载 `shoplifting.pt`(文件缺失)。提供基于 Ultralytics 微调 `yolov11n.pt` 的训练脚本 + `data.yaml` 模板 + 公开零售盗窃数据集建议;训练产物放入后现有后端自动启用。
9. **可选对比**:yolov8n-pose → yolov8s-pose,在评估集上量化精度/速度取舍。

## 执行顺序与人工配合点

- 阶段 1 的脚本写完后,**需要你人工标注 CSV**(约 31 条,预计 30–60 分钟);查全评估视频也需你提供。
- 阶段 2 中第 4 步(修复聚合路径)不依赖标注,可立即做。
- 验收标准:调优后 eval 脚本的 F1 报告对比基线有可量化提升。