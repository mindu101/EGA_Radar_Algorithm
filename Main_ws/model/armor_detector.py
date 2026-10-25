from model.armor_light_classifier import ArmorLightClassifier
from model.digit_classifier.predictor import DigitClassifier
import PIL
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


class ArmorClassifier:
    def __init__(self, armor_params, digit_model_type, digit_weights_path, debug=False):
        """
        Initialize the ArmorClassifier with armor parameters and digit classifier settings.

        Args:
            armor_params: Parameters for armor plate detection.
            digit_model_type: Type of the digit classifier model (e.g., 'resnet18', 'efficientnet', 'mobilenet').
            digit_weights_path: Path to the weights file for the digit classifier.
            debug: Whether to enable debug mode for additional logging and visualization.
        """
        self.armor_light_classifier = ArmorLightClassifier(armor_params, debug=debug)
        self.digit_classifier = DigitClassifier(
            model_type=digit_model_type,
            weights_path=digit_weights_path,
        )
        self.debug = debug
        self.verbose = bool(debug)

    def classify(self, img):
        """
        Classify the armor plate in the given image.

        Args:
            img: Input image containing the armor plate.

        Returns:
            A dictionary with detected lights and their classifications.
        """
        # Step 1: Preprocess the image to get binary representation
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        color_result = self.armor_light_classifier.classify_color(bgr_img)
        # Step 2: Classify the lights based on color
        pil_iamge = PIL.Image.fromarray(img).convert("RGB")
        armor_id, conf = self.digit_classifier.predict(pil_iamge, return_names=False)

        if color_result == "RED":
            armor_id += 5
        elif color_result == "GREY":
            armor_id += 10

        armor_name = class_names[armor_id]
        return armor_name, armor_id

    def classify_batch(self, imgs):
        """
        Classify a batch of images containing armor plates.

        Args:
            imgs: List of input images containing armor plates.

        Returns:
            A list of dictionaries with detected lights and their classifications.
        """
        import time

        start = time.time()
        pattern_indxes, confs = self.digit_classifier.predict_batch(
            [PIL.Image.fromarray(img).convert("RGB") for img in imgs],
            return_names=False,
        )
        if self.verbose:
            print(f"[Digit Classifier]: {time.time() - start:.4f} seconds")
        start = time.time()
        max_confs = list(map(max, confs))

        colors = []
        for img in imgs:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            color_result = self.armor_light_classifier.classify_color(bgr_img)
            colors.append(color_result)

        if self.verbose:
            print(f"[Find lights]: {time.time() - start:.4f} seconds")
        armor_names, armor_ids = [], []
        for idx, color in zip(pattern_indxes, colors):
            if color == "RED":
                idx += 5
            elif color == "GREY":
                idx += 10
            armor_names.append(class_names[idx])
            armor_ids.append(idx)

        return armor_names, armor_ids, max_confs


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


class ArmorDetectorClassifier:
    def __init__(self, armor_detector_model, armor_classifier_model, verbose=False):
        self.armor_detector_model = armor_detector_model
        self.armor_classifier_model = armor_classifier_model
        self.verbose = bool(verbose)

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
        # Step 1: 使用原始armor detector进行检测
        import time

        start = time.time()
        raw_detections_list, annotated_images_list = (
            self.armor_detector_model.predict_batch(imgs)
        )
        if self.verbose:
            print(f"[Armor Detections]: {time.time() - start:.4f}seconds")

        # Step 2: 收集所有装甲板区域
        all_armor_crops = []
        all_detection_info = []  # 存储 (img_idx, xyxy, confidence)

        for img_idx, (img, raw_detections) in enumerate(zip(imgs, raw_detections_list)):
            if not raw_detections:  # 如果没有检测到装甲板
                continue

            # 提取装甲板区域
            for detection in raw_detections:
                original_class_id, xyxy, confidence = detection
                x1, y1, x2, y2 = map(int, xyxy)

                # 裁剪装甲板区域
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:  # 确保裁剪区域有效
                    all_armor_crops.append(crop)
                    all_detection_info.append((img_idx, xyxy, confidence))

        # Step 3: 一起分类所有装甲板
        enhanced_detections_list = [[] for _ in imgs]  # 初始化结果列表

        if all_armor_crops:
            # 批量分类所有装甲板
            armor_names, armor_ids, digit_confs = (
                self.armor_classifier_model.classify_batch(all_armor_crops)
            )

            # Step 4: 将分类结果分配回对应的图像
            for (img_idx, xyxy, original_conf), new_class_id, digit_conf in zip(
                all_detection_info, armor_ids, digit_confs
            ):
                enhanced_detections_list[img_idx].append(
                    (
                        int(new_class_id),
                        xyxy,
                        float(original_conf),
                        float(digit_conf),
                        int(new_class_id % 5),
                    )
                )

        return enhanced_detections_list, annotated_images_list

    def predict(self, img):
        """
        单张图像预测，保持接口一致性
        """
        detections_list, annotated_images_list = self.predict_batch([img])
        return detections_list[0], annotated_images_list[0]
