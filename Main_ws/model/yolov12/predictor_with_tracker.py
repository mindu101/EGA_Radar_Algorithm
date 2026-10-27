import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
from types import MethodType

from model.yolov12.image_utils import rgb_to_ultralytics_bgr_view


class PredictorWithTracker:

    def __init__(
        self,
        model_path: str,
        img_size=1280,
        conf_thres=0.05,
        iou_thres=0.5,
        device="cuda",
        max_det=10,
        tracker_config_path: str = "config/bytetrack.yaml",
        visualize=False,
        verbose=False,
    ):
        self.model_path = model_path
        self.input_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        
        self.visualize = visualize
        self.verbose = bool(verbose)
        self.max_det = max_det
        self.model = YOLO(model_path, task="detect")
        # self.device = self.model.device
        print(self.model.names)
        print(f"Using device: {self.device}")
        print(
            "[YOLO] "
            f"weights={model_path} imgsz={img_size} conf={conf_thres:.3f} "
            f"ultralytics={ultralytics.__version__} "
            f"torch={torch.__version__} cuda={torch.version.cuda}"
        )
        # self.model = self.model.to(self.device)
        # self.model.eval()
        self.tracker_config_path = tracker_config_path
        self.last_timing_stats = {}
        self._timing_hooks_installed = False
        self._last_tracker_callback_s = 0.0
        self._last_botsort_update_s = 0.0
        self.last_result_box_count = 0
        self.last_id_box_count = 0
        self.last_raw_yolo_boxes = ()

    def _record_timing(self, name: str, duration_s: float):
        """记录最近一次车辆检测内部阶段耗时，单位为秒。"""
        self.last_timing_stats[name] = float(duration_s)

    def _install_tracker_timing_hooks(self):
        """包裹 Ultralytics 跟踪回调和 BOTSORT.update，不修改第三方源码。"""
        if self._timing_hooks_installed:
            return
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if not trackers:
            return

        callbacks = self.model.callbacks.get("on_predict_postprocess_end", [])
        tracker_callback_index = None
        for index, callback in enumerate(callbacks):
            raw_callback = getattr(callback, "func", callback)
            if (
                getattr(raw_callback, "__module__", "")
                == "ultralytics.trackers.track"
                and getattr(raw_callback, "__name__", "")
                == "on_predict_postprocess_end"
            ):
                tracker_callback_index = index
                break
        if tracker_callback_index is None:
            return

        original_callback = callbacks[tracker_callback_index]

        def timed_tracker_callback(callback_predictor):
            started = time.perf_counter()
            try:
                # Preserve post-NMS detector boxes before Ultralytics lets the
                # tracker replace or empty Results.boxes.
                raw_boxes = []
                try:
                    for result in getattr(callback_predictor, "results", ()):
                        boxes = getattr(result, "boxes", None)
                        if boxes is None or len(boxes) == 0:
                            continue
                        packed = torch.cat(
                            (boxes.xyxy, boxes.conf[:, None]), dim=1
                        ).detach().cpu().numpy()
                        raw_boxes.extend(
                            tuple(float(value) for value in row)
                            for row in packed
                        )
                    self.last_raw_yolo_boxes = tuple(raw_boxes)
                except Exception:
                    self.last_raw_yolo_boxes = ()
                return original_callback(callback_predictor)
            finally:
                self._last_tracker_callback_s += time.perf_counter() - started

        callbacks[tracker_callback_index] = timed_tracker_callback

        for tracker in trackers:
            if getattr(tracker, "_rm_timing_wrapped", False):
                continue
            original_update = tracker.update

            def timed_update(_tracker, *args, _original=original_update, **kwargs):
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self._last_botsort_update_s += time.perf_counter() - started

            tracker.update = MethodType(timed_update, tracker)
            tracker._rm_timing_wrapped = True
        self._timing_hooks_installed = True

    def _record_model_track_breakdown(self, result, track_wall_s: float):
        """记录 Ultralytics 官方分段计时和项目侧 BoT-SORT 精确计时。"""
        speed = getattr(result, "speed", {}) or {}
        preprocess_s = float(speed.get("preprocess") or 0.0) / 1000.0
        inference_s = float(speed.get("inference") or 0.0) / 1000.0
        postprocess_s = float(speed.get("postprocess") or 0.0) / 1000.0
        callback_s = max(0.0, float(self._last_tracker_callback_s))
        botsort_s = max(0.0, float(self._last_botsort_update_s))
        callback_io_s = max(0.0, callback_s - botsort_s)
        framework_s = max(
            0.0,
            float(track_wall_s)
            - preprocess_s
            - inference_s
            - postprocess_s
            - callback_s,
        )
        self._record_timing("03.2 model.track总时间（模块03.2总时间）", track_wall_s)
        self._record_timing("03.2.1 Ultralytics内部预处理（模块03.2.1总时间）", preprocess_s)
        self._record_timing("03.2.2 车辆模型推理（原子模块03.2.2）", inference_s)
        self._record_timing("03.2.3 YOLO后处理（模块03.2.3总时间）", postprocess_s)
        self._record_timing("03.2.4 BoT-SORT回调（模块03.2.4总时间）", callback_s)
        self._record_timing("03.2.4.1 BoT-SORT跟踪更新（原子模块03.2.4.1）", botsort_s)
        self._record_timing(
            "03.2.4.2 跟踪输入转换与结果写回（模块03.2.4.2总时间）",
            callback_io_s,
        )
        self._record_timing("03.2.5 Ultralytics框架调度余量（原子模块03.2.5）", framework_s)

    def predict(self, image):
        self.last_timing_stats = {}
        self._last_tracker_callback_s = 0.0
        self._last_botsort_update_s = 0.0
        self.last_result_box_count = 0
        self.last_id_box_count = 0
        self.last_raw_yolo_boxes = ()
        predict_start = time.perf_counter()
        detections = []
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            print("[Warning]: Invalid or empty image provided.")
            return detections, None

        stage_start = time.perf_counter()
        # 保留原图长宽比，让 Ultralytics 在 imgsz 下执行 letterbox 预处理。
        # 输入仍是项目内部 RGB；这里仅创建 BGR 通道视图，不复制原始图像。
        ultralytics_img = rgb_to_ultralytics_bgr_view(image)
        self._record_timing(
            "03.1.1 原图RGB转BGR共享视图（原子模块03.1.1）",
            time.perf_counter() - stage_start,
        )
        stage_start = time.perf_counter()
        results = self.model.track(
            ultralytics_img,
            stream=False,
            persist=True,
            tracker=self.tracker_config_path,
            imgsz=self.input_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            verbose=self.verbose,
        )[0]
        track_wall_s = time.perf_counter() - stage_start
        self._record_model_track_breakdown(results, track_wall_s)
        # 第一次调用创建 tracker；十次预热保证正式运行前 hook 已生效。
        self._install_tracker_timing_hooks()

        stage_start = time.perf_counter()
        if results.boxes is None or len(results.boxes) == 0:
            self.last_result_box_count = 0
            self.last_id_box_count = 0
            self._record_timing(
                "03.3.1 跟踪结果张量合并与CPU转换（原子模块03.3.1）",
                0.0,
            )
            self._record_timing(
                "03.3.2 坐标还原与Python结果组装（原子模块03.3.2）",
                0.0,
            )
            self._record_timing("03.3 车辆结果解析（模块03.3总时间）", time.perf_counter() - stage_start)
            self._record_timing("03 车辆检测与BoT-SORT（模块03总时间）", time.perf_counter() - predict_start)
            if self.verbose:
                print(f"[Detect car time]: {self.last_timing_stats['03 车辆检测与BoT-SORT（模块03总时间）']:.4f} seconds")
            return detections, None


        boxes = results.boxes
        self.last_result_box_count = len(boxes)
        self.last_id_box_count = 0 if boxes.id is None else len(boxes.id)
        if boxes.id is None:
            if self.verbose:
                print("[Warning]: Tracker has not assigned IDs to current boxes.")
            self._record_timing(
                "03.3.1 跟踪结果张量合并与CPU转换（原子模块03.3.1）",
                0.0,
            )
            self._record_timing(
                "03.3.2 坐标还原与Python结果组装（原子模块03.3.2）",
                0.0,
            )
            self._record_timing("03.3 车辆结果解析（模块03.3总时间）", time.perf_counter() - stage_start)
            self._record_timing("03 车辆检测与BoT-SORT（模块03总时间）", time.perf_counter() - predict_start)
            return detections, None

        # 所有框一次性回传 CPU，避免每个框的 conf/cls/xyxy/id 分别同步 CUDA。
        pack_started = time.perf_counter()
        packed = torch.cat(
            (
                boxes.xyxyn,
                boxes.conf[:, None],
                boxes.cls[:, None],
                boxes.id[:, None],
            ),
            dim=1,
        ).detach().cpu().numpy()
        self._record_timing(
            "03.3.1 跟踪结果张量合并与CPU转换（原子模块03.3.1）",
            time.perf_counter() - pack_started,
        )
        convert_started = time.perf_counter()
        packed[:, [0, 2]] *= image.shape[1]
        packed[:, [1, 3]] *= image.shape[0]
        for x1, y1, x2, y2, conf, cls, track_id in packed:
            detections.append(
                (
                    int(cls),
                    [int(x1), int(y1), int(x2), int(y2)],
                    float(conf),
                    int(track_id),
                )
            )
        self._record_timing(
            "03.3.2 坐标还原与Python结果组装（原子模块03.3.2）",
            time.perf_counter() - convert_started,
        )
        self._record_timing("03.3 车辆结果解析（模块03.3总时间）", time.perf_counter() - stage_start)
        self._record_timing("03 车辆检测与BoT-SORT（模块03总时间）", time.perf_counter() - predict_start)
        if self.verbose:
            print(f"[Detect car time]: {self.last_timing_stats['03 车辆检测与BoT-SORT（模块03总时间）']:.4f} seconds")
        return detections, None

    def visualize_results(self, image, detections, class_names=None):
        """可视化检测结果"""
        # vis_image = image.copy()
        # vis_image = cv2.resize(
        #     image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        # )
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # 默认类别名称
        if class_names is None:
            class_names = {0: "car"}  # 根据你的模型调整

        # 颜色列表
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for detection in detections:
            print(detection)
            cls_id, bbox, conf, track_id = detection
            x1, y1, x2, y2 = bbox

            # 转换归一化坐标到像素坐标
            # x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

            # 选择颜色
            color = colors[cls_id % len(colors)]

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            label = f"{class_names.get(cls_id, f'Class_{cls_id}')}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return vis_image


if __name__ == "__main__":
    import cv2
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    predictor = PredictorWithTracker(
        model_path=str(project_root / "weights/car/best.engine"),
        visualize=True,
    )
    image_path = str(project_root / "demo/demo1.jpg")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 预测
    results = predictor.predict(image)
    print("Detection result:", results)

    # 可视化
    if predictor.visualize:
        # 可视化结果
        vis_image = predictor.visualize_results(image, results[0])
        save_path = "visualize/yolov12/zk_5.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
