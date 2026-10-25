from model.digit_classifier.predictor import DigitClassifier
import cv2
import time


class_names = [
    "B1",
    "B2",
    "B3",
    "B4",
    "BS",
    "R1",
    "R2",
    "R3",
    "R4",
    "RS",
    "G1",
    "G2",
    "G3",
    "G4",
    "GS",  # dead
]


class TwoStepArmorDetectorClassifier:
    """A two-step armor detector and classifier that first detects armor plates
    1. The first step uses an armor detector to find the positions of armor plates in the image. Also returns the colors.
    2. The second step crops the detected armor plates and classifies them using a digit classifier.
    """

    def __init__(self, armor_detector_model, digit_classifier_model, verbose=False):
        """
        Initialize the TwoStepArmorDetectorClassifier with armor detector and digit classifier models.

        Args:
            armor_detector_model: The model used for detecting armor plates.
            digit_classifier_model: The model used for classifying the detected armor plates.
        """
        self.armor_detector_model = armor_detector_model
        self.digit_classifier = digit_classifier_model
        self.verbose = bool(verbose)
        self.last_timing_stats = {}

    @staticmethod
    def _compose_armor_class_id(color_id: int, pattern_id: int):
        """按颜色模型类别与数字类别生成统一装甲板 class_id。

        颜色模型输出约定为 0=死亡/灰色、1=红色、2=蓝色；统一 class_id
        约定为 0~4=蓝方、5~9=红方、10~14=死亡/灰色。
        """
        color_offsets = {
            0: 10,
            1: 5,
            2: 0,
        }
        offset = color_offsets.get(int(color_id))
        if offset is None or not 0 <= int(pattern_id) < 5:
            return None
        return int(pattern_id) + offset

    @classmethod
    def from_config(cls, config):
        from model.yolov12.predictor import Predictor

        original_armor_detector = Predictor(
            model_path=config["armor_detector"]["weights_path"],
            img_size=config["armor_detector"]["img_size"],
            max_det=config["armor_detector"]["max_det"],
            conf_thres=config["armor_detector"]["conf_thres"],
            iou_thres=config["armor_detector"]["iou_thres"],
            verbose=config["armor_detector"].get("verbose", False),
        )
        digit_classifier = DigitClassifier(
            model_type=config["armor_detector"]["digit_model_type"],
            weights_path=config["armor_detector"]["digit_weights_path"],
        )

        armor_detector = TwoStepArmorDetectorClassifier(
            armor_detector_model=original_armor_detector,
            digit_classifier_model=digit_classifier,
            verbose=config["armor_detector"].get("verbose", False),
        )
        return armor_detector

    def predict_batch(self, imgs):
        """
        保持与YOLOv5Detector相同的接口

        Args:
            imgs: List of numpy arrays (images)

        Returns:
            tuple: (detections_list, annotated_images_list)
                - detections_list: List[List[tuple(class_id, xyxy, confidence)]]
                - annotated_images_list: List of annotated images
        """
        self.last_timing_stats = {}
        total_started = time.perf_counter()
        started = time.perf_counter()
        raw_detections_list, annotated_images_list = (
            self.armor_detector_model.predict_batch(imgs)
        )
        self.last_timing_stats["装甲板位置神经网络推理与结果回传"] = (
            time.perf_counter() - started
        )
        for timing_name, timing_value in getattr(
            self.armor_detector_model, "last_timing_stats", {}
        ).items():
            self.last_timing_stats[timing_name] = float(timing_value)
        if self.verbose:
            print(
                "Time for armor position detection: "
                f"{self.last_timing_stats['装甲板位置神经网络推理与结果回传']:.4f} seconds"
            )

        # Step 2: 收集所有装甲板区域
        started = time.perf_counter()
        all_armor_crops = []
        all_detection_info = []

        for img_idx, (img, raw_detections) in enumerate(zip(imgs, raw_detections_list)):
            if not raw_detections:  # 如果没有检测到装甲板
                continue

            # 提取装甲板区域
            for detection in raw_detections:
                color_id, xyxy, confidence = detection
                x1, y1, x2, y2 = map(int, xyxy)

                # 裁剪装甲板区域
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:  # 确保裁剪区域有效
                    all_armor_crops.append(crop)
                    all_detection_info.append((img_idx, xyxy, confidence, color_id))
        self.last_timing_stats["07.1 装甲板ROI裁边与切片（原子模块07.1）"] = time.perf_counter() - started

        # Step 3: 一起分类所有装甲板
        enhanced_detections_list = [[] for _ in imgs]  # 初始化结果列表
        if all_armor_crops:
            # 批量分类所有装甲板
            # numpy HWC RGB 直传：DigitClassifier 检测到 numpy 输入时走
            # cv2 批量预处理（preprocess_numpy_batch），省掉逐张 PIL 四次拷贝
            started = time.perf_counter()
            digit_inputs = all_armor_crops
            self.last_timing_stats["07.2 装甲ROI直传数字分类器（原子模块07.2）"] = (
                time.perf_counter() - started
            )
            started = time.perf_counter()
            pattern_indexes, digit_confs = self.digit_classifier.predict_batch(
                digit_inputs,
                return_names=False,
            )
            self.last_timing_stats["数字神经网络推理与结果回传"] = (
                time.perf_counter() - started
            )
            for timing_name, timing_value in getattr(
                self.digit_classifier, "last_timing_stats", {}
            ).items():
                self.last_timing_stats[timing_name] = float(timing_value)
            started = time.perf_counter()
            for (img_idx, xyxy, yolo_conf, color_id), pattern_id, digit_conf in zip(
                all_detection_info, pattern_indexes, digit_confs
            ):
                if pattern_id == 5:  # 前哨站不进入机器人身份候选
                    continue
                armor_id = self._compose_armor_class_id(color_id, pattern_id)
                if armor_id is None:
                    continue

                max_digit_conf = max(digit_conf)
                enhanced_detections_list[img_idx].append(
                    (
                        armor_id,
                        xyxy,
                        float(yolo_conf),
                        float(max_digit_conf),
                        int(pattern_id),
                    )
                )
            self.last_timing_stats["09.1 装甲颜色数字合成与结果分配（原子模块09.1）"] = (
                time.perf_counter() - started
            )
        else:
            self.last_timing_stats["07.2 装甲ROI直传数字分类器（原子模块07.2）"] = 0.0
            self.last_timing_stats["数字神经网络推理与结果回传"] = 0.0
            self.last_timing_stats["09.1 装甲颜色数字合成与结果分配（原子模块09.1）"] = 0.0
        self.last_timing_stats["装甲与数字识别阶段总时间"] = (
            time.perf_counter() - total_started
        )
        return enhanced_detections_list, annotated_images_list

