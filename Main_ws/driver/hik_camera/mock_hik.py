import cv2
from threading import Lock, Thread
import time


class SimpleHikCamera:
    _instance = None

    def __init__(self, video_source: str):
        """
        Initialize the camera with the given video source.

        Args:
            video_source (int or str): The video source, can be an integer for webcam or a string for video file.
        """


        self.video_source = video_source
        self.capture = cv2.VideoCapture(video_source)
        self.__class__._instance = self

        self.cur_image = None
        self.cur_time_stamp = None
        self.lock = Lock()
        self._fps = 0.0
        self._last_frame_time = None
        print(f"Initialized camera with source: {video_source}")
        self.fetch_loop = Thread(target=self.fetch_image_loop, daemon=True)
        self.fetch_loop.start()

    def fetch_image_loop(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                now = time.time()
                if self._last_frame_time is not None:
                    dt = now - self._last_frame_time
                    if dt > 1e-6:
                        self._fps = 0.8 * self._fps + 0.2 / dt
                self._last_frame_time = now
                with self.lock:
                    self.cur_image = frame
                    self.cur_time_stamp = now
            else:
                print("Failed to capture image from camera.")
                break
            time.sleep(0.033)

        self.capture.release()

    def get_image_latest(self, group_id: str = "", timeout=1):
        with self.lock:
            image = self.cur_image
            stamp = self.cur_time_stamp
        if image is None:
            print("No image captured yet.")
            return None, 0.0
        # 帧发布后保持只读，消费者共享引用，避免每次取图复制整帧。
        return image, stamp

    def register_group(self, group_id: str):
        """
        Register a group ID for the camera.

        Args:
            group_id (str): The group ID to register.
        """
        print(f"Registered group ID: {group_id}")
    
    def start_streaming(self):
        pass
    
    def stop_streaming(self):
        pass

    def get_fps(self) -> float:
        return self._fps

    def get_time(self) -> float:
        return time.time()

    def is_connected(self) -> bool:
        if self._last_frame_time is None:
            return False
        return (time.time() - self._last_frame_time) < 1.0
