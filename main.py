import argparse
import logging
import os
import queue
import sys
import threading
import time
from queue import Queue

import cv2
import numpy as np

os.makedirs("log", exist_ok=True)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s", handlers=[
    logging.FileHandler("log/file.log", encoding="utf-8"),
    logging.StreamHandler(sys.stdout)
])

logger = logging.getLogger(__name__)


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    '''Sensor X'''

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, name, width: int, height: int):
        self.width = width
        self.height = height
        self.name = name
        self.camera = cv2.VideoCapture(name)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get(self):
        if not self.camera.isOpened():
            logger.error("Error opening stream... trying to open")
            try:
                self.camera.open(self.name)
            except Exception as error:
                logger.error(f"Can`t open stream: {error}")
                return None
        try:
            ret, frame = self.camera.read()
            if ret:
                return frame
            else:
                logger.error(f"Incorrect reading frame")
                return None
        except Exception as error:
            logger.error(f"Error reading frame: {error}")
        return None

    def __del__(self):
        self.camera.release()


class WindowImage:
    def __init__(self, window_name="WindowImage", fps: int = 60):
        self.delay = 1 / fps
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, image):
        time.sleep(self.delay)
        cv2.imshow(self.window_name, image)

    def __del__(self):
        cv2.destroyWindow(self.window_name)


def window_func(fps: int, frame_queue: Queue, sensor1_queue: Queue, sensor2_queue: Queue, sensor3_queue: Queue):
    frame = np.random.rand(128, 128, 3)
    logger.debug("Starting window thread")
    window = WindowImage(fps=fps)
    sensor1_res = None
    sensor2_res = None
    sensor3_res = None
    while True:
        key = cv2.waitKey(1) & 0xFF
        if chr(key) == "q":
            logger.info("'Q' pressed. Terminating window...")
            break

        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sensor1_res = sensor1_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sensor2_res = sensor2_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sensor3_res = sensor3_queue.get_nowait()
        except queue.Empty:
            pass

        cv2.rectangle(frame, (0, 0), (130, 70), (55, 81, 229), -1)
        cv2.putText(frame, f"Sensor1 {sensor1_res}", (10, 20), 1,
                    1, (0, 0, 0), 2)
        cv2.putText(frame, f"Sensor2 {sensor2_res}", (10, 40), 1,
                    1, (0, 0, 0), 2)
        cv2.putText(frame, f"Sensor3 {sensor3_res}", (10, 60), 1,
                    1, (0, 0, 0), 2)
        if frame is None:
            break
        window.show(frame)

    logger.debug("Ending window thread")


def camera_func(camera: Sensor, sensor_queue: Queue, main_thread_works):
    logger.debug("Starting camera thread")
    while main_thread_works:
        frame = camera.get()
        try:
            sensor_queue.put_nowait(frame)
        except queue.Full:
            pass
        if frame is None:
            break
    logger.debug("Ending camera thread")


def sensor_func(sensor: Sensor, sensor_queue: Queue, main_thread_works):
    while main_thread_works:
        result = sensor.get()
        try:
            sensor_queue.put_nowait(result)
        except queue.Full:
            pass


if __name__ == "__main__":
    cv2.startWindowThread()
    parser = argparse.ArgumentParser(
        prog='Camera with sensors',
        description='Program starts streaming camera and shows sensor data.',
        epilog='All parameters are optional')
    parser.add_argument('-n', '--name', help="??camera name??")  # positional argument
    parser.add_argument('-s', '--size', help="size in pixel: WIDTHxHEIGHT. 1280x720 by default.")  # option that takes a value
    parser.add_argument('-f', '--fps', help="frame rate value")
    args = parser.parse_args()
    logger.debug(f"--name {args.name} --size {args.size} --fps {args.fps}")
    name = 1
    w, h = 640, 480
    fps = 30
    if args.name is not None:
        name = args.name
    if args.size is not None:
        h, w = map(int, args.size.split('x'))
    if args.fps is not None:
        args.fps = int(args.fps)

    camera = SensorCam(name=name, width=w, height=h)

    sensor1 = SensorX(0.01)
    sensor2 = SensorX(0.1)
    sensor3 = SensorX(1)
    frame_queue = Queue(maxsize=2)
    sensor1_queue = Queue(maxsize=2)
    sensor2_queue = Queue(maxsize=2)
    sensor3_queue = Queue(maxsize=2)
    main_thread_works = True
    camera_thread = threading.Thread(target=camera_func, args=(camera, frame_queue, main_thread_works,))

    sensor_thread1 = threading.Thread(target=sensor_func, args=(sensor1, sensor1_queue, main_thread_works,))

    sensor_thread2 = threading.Thread(target=sensor_func, args=(sensor2, sensor2_queue, main_thread_works,))

    sensor_thread3 = threading.Thread(target=sensor_func, args=(sensor3, sensor3_queue, main_thread_works,))

    # window_thread = threading.Thread(target=window_func, args=(fps, frame_queue, sensor1_queue, sensor2_queue, sensor3_queue,))
    camera_thread.start()
    sensor_thread3.start()
    sensor_thread2.start()
    sensor_thread1.start()

    frame = np.random.rand(128, 128, 3)
    logger.debug("Starting window thread")
    window = WindowImage(fps=fps)
    sensor1_res = None
    sensor2_res = None
    sensor3_res = None
    while True:
        key = cv2.waitKey(1) & 0xFF
        if chr(key) == "q":
            logger.info("'Q' pressed. Terminating window...")
            break

        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sensor1_res = sensor1_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sensor2_res = sensor2_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sensor3_res = sensor3_queue.get_nowait()
        except queue.Empty:
            pass

        cv2.rectangle(frame, (0, 0), (130, 70), (55, 81, 229), -1)
        cv2.putText(frame, f"Sensor1 {sensor1_res}", (10, 20), 1,
                    1, (0, 0, 0), 2)
        cv2.putText(frame, f"Sensor2 {sensor2_res}", (10, 40), 1,
                    1, (0, 0, 0), 2)
        cv2.putText(frame, f"Sensor3 {sensor3_res}", (10, 60), 1,
                    1, (0, 0, 0), 2)
        if frame is None:
            main_thread_works = False
            break
        window.show(frame)
    window.__del__()
    logger.debug("Ending window thread")
    camera_thread.join()
    sensor_thread1.join()
    sensor_thread2.join()
    sensor_thread3.join()

