from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time

from model.yolov12.image_utils import rgb_to_ultralytics_bgr_view


class Predictor:

    def __init__(
        self,
        model_path: str,
        img_size=640,
        conf_thres=0.15,
        iou_thres=0.45,
        max_det=10,
        device="cuda",
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
        self.last_timing_stats = {}

        self.model = YOLO(model_path, task='detect')
        print('**********Predictor**********')
        print("Model loaded,model_path:",model_path)
        print(self.model.names)
        print(f"Using device: {self.device}")

        if model_path.split(".")[-1] != "engine":
            self.model = self.model.to(self.device)
            self.model.eval()

    def predict(self, image):
        detections = []
        # 项目内部统一为 RGB；Ultralytics 的 NumPy 接口约定输入为 BGR。
        # 仅创建通道反转视图，避免额外复制图像。
        ultralytics_image = rgb_to_ultralytics_bgr_view(image)
        results = self.model.predict(
            ultralytics_image,
            imgsz=self.input_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            verbose=self.verbose,
        )
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            packed = torch.cat(
                (boxes.xyxyn, boxes.conf[:, None], boxes.cls[:, None]), dim=1
            ).detach().cpu().numpy()
            packed[:, [0, 2]] *= image.shape[1]
            packed[:, [1, 3]] *= image.shape[0]
            for x1, y1, x2, y2, conf, cls in packed:
                detections.append(
                    (
                        int(cls),
                        [int(x1), int(y1), int(x2), int(y2)],
                        float(conf),
                    )
                )
        return detections, None

    def predict_batch(self, images):
        self.last_timing_stats = {}
        total_started = time.perf_counter()
        all_detections = [[] for _ in images]
        # print(images)
        started = time.perf_counter()
        ultralytics_images = [rgb_to_ultralytics_bgr_view(image) for image in images]
        self.last_timing_stats[
            "06.1 装甲车辆ROI转BGR共享视图（原子模块06.1）"
        ] = time.perf_counter() - started
        started = time.perf_counter()
        results = self.model.predict(
            ultralytics_images,
            imgsz=self.input_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            verbose=self.verbose,
        )
        model_wall_s = time.perf_counter() - started
        batch_size = max(1, len(results))
        speed = getattr(results[0], "speed", {}) if results else {}
        preprocess_s = float(speed.get("preprocess") or 0.0) * batch_size / 1000.0
        inference_s = float(speed.get("inference") or 0.0) * batch_size / 1000.0
        postprocess_s = float(speed.get("postprocess") or 0.0) * batch_size / 1000.0
        self.last_timing_stats["06.2 Ultralytics装甲内部预处理（模块06.2总时间）"] = preprocess_s
        self.last_timing_stats["06.3 装甲板位置神经网络推理（原子模块06.3）"] = inference_s
        self.last_timing_stats["06.4 装甲YOLO后处理（模块06.4总时间）"] = postprocess_s
        self.last_timing_stats["06.5 装甲Ultralytics框架调度余量（原子模块06.5）"] = max(
            0.0, model_wall_s - preprocess_s - inference_s - postprocess_s
        )
        self.last_timing_stats["06.2-06.5 装甲model.predict总时间（模块06.2至06.5总时间）"] = model_wall_s
        started = time.perf_counter()
        batch_tensors = []
        batch_counts = []
        for result in results:
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            batch_counts.append(count)
            if count > 0:
                batch_tensors.append(
                    torch.cat(
                        (boxes.xyxyn, boxes.conf[:, None], boxes.cls[:, None]),
                        dim=1,
                    )
                )

        # 不同车辆 ROI 的装甲结果合并后只进行一次 GPU→CPU 复制。
        if batch_tensors:
            packed_all = torch.cat(batch_tensors, dim=0).detach().cpu().numpy()
        else:
            packed_all = np.empty((0, 6), dtype=np.float32)

        offset = 0
        for image_index, (image, count) in enumerate(zip(images, batch_counts)):
            packed = packed_all[offset : offset + count].copy()
            offset += count
            if count == 0:
                continue
            packed[:, [0, 2]] *= image.shape[1]
            packed[:, [1, 3]] *= image.shape[0]
            all_detections[image_index] = [
                (
                    int(cls),
                    [int(x1), int(y1), int(x2), int(y2)],
                    float(conf),
                )
                for x1, y1, x2, y2, conf, cls in packed
            ]
        self.last_timing_stats[
            "06.6 装甲结果批量CPU回传与坐标组装（原子模块06.6）"
        ] = time.perf_counter() - started
        self.last_timing_stats["06 装甲位置检测（模块06总时间）"] = (
            time.perf_counter() - total_started
        )
        return all_detections, None

    def visualize_results(self, image, detections, class_names=None):
        """可视化检测结果"""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # 默认类别名称
        if class_names is None:
            class_names = {0: "car", 1: "person", 2: "bike"}  # 根据你的模型调整

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
            print("[predictor.py]visualize_results--detection:", detection)
            cls_id, bbox, conf = detection
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

    predictor = Predictor(model_path="weights/armor/armor_best.engine", visualize=True)
    image_path = Path(__file__).resolve().parents[2] / "demo/demo1.jpg"
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pred = cv2.resize(image, (640, 640))
    # 预测
    results = predictor.predict(image_pred) #预测结果已经是像素坐标
    print("检测结果:", results)

    # 可视化
    if predictor.visualize:
        # 可视化结果
        vis_image = predictor.visualize_results(image, results[0])
        save_path = "visualize/yolov12/0011.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
