import time
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import psutil
import numpy as np
from picamera2 import Picamera2

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


HEF_PATH = Path("weights/espcn_x2_finetuned_224_fixed.hef")

HOST = "0.0.0.0"
PORT = 8080

CAMERA_SIZE = (224, 224)
JPEG_QUALITY = 100
PRINT_EVERY = 30

latest_jpeg = None
latest_lock = threading.Lock()
running = True


def get_hw_from_shape(shape):
    dims = list(shape)

    if len(dims) == 3:
        return int(dims[0]), int(dims[1])

    if len(dims) == 4:
        return int(dims[1]), int(dims[2])

    raise RuntimeError(f"Unsupported shape: {shape}")


def preprocess_frame(frame_rgb, lr_w, lr_h):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    if gray.shape != (lr_h, lr_w):
        gray = cv2.resize(
            gray,
            (lr_w, lr_h),
            interpolation=cv2.INTER_AREA,
        )

    x = gray.astype(np.float32) / 255.0
    x = x[np.newaxis, :, :, np.newaxis]  # NHWC

    return gray, x


def postprocess(output):
    sr = np.squeeze(output)
    sr = np.clip(sr, 0.0, 1.0)
    sr = (sr * 255.0).round().astype(np.uint8)

    return sr


def make_display(lr, sr):
    sr_h, sr_w = sr.shape

    bicubic = cv2.resize(
        lr,
        (sr_w, sr_h),
        interpolation=cv2.INTER_CUBIC,
    )

    bicubic = cv2.cvtColor(bicubic, cv2.COLOR_GRAY2BGR)
    sr_vis = cv2.cvtColor(sr, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        bicubic,
        "Bicubic x2",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        sr_vis,
        "Hailo ESPCN x2",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    return np.hstack((bicubic, sr_vis))


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
                <html>
                <head>
                    <title>Hailo ESPCN Live Demo</title>
                </head>
                <body style="background:#111;color:white;font-family:sans-serif;">
                    <h2>Hailo ESPCN Live Demo</h2>
                    <img src="/stream.mjpg" style="max-width:100%;height:auto;">
                </body>
                </html>
                """
            )
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.end_headers()

        global latest_jpeg, running

        while running:
            with latest_lock:
                frame = latest_jpeg

            if frame is not None:
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except BrokenPipeError:
                    break

            time.sleep(0.01)

    def log_message(self, format, *args):
        return


def start_server():
    server = HTTPServer((HOST, PORT), MJPEGHandler)
    print(f"Open this on your laptop: http://rpi5-longobardi.local:{PORT}")
    server.serve_forever()


def main():
    global latest_jpeg, running

    process = psutil.Process()

    server_thread = threading.Thread(
        target=start_server,
        daemon=True,
    )
    server_thread.start()

    hef = HEF(str(HEF_PATH))

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )

        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        output_vstreams_params = OutputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        input_info = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]

        input_name = input_info.name
        output_name = output_info.name

        lr_h, lr_w = get_hw_from_shape(input_info.shape)
        hr_h, hr_w = get_hw_from_shape(output_info.shape)

        print("Input:", input_name, input_info.shape)
        print("Output:", output_name, output_info.shape)
        print(f"Resolution: {lr_w}x{lr_h} -> {hr_w}x{hr_h}")

        if (lr_w, lr_h) != CAMERA_SIZE:
            print(
                f"Warning: CAMERA_SIZE={CAMERA_SIZE}, "
                f"but HEF input is {lr_w}x{lr_h}. "
                f"Resize will still be applied."
            )

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={
                "size": CAMERA_SIZE,
                "format": "RGB888",
            }
        )
        picam2.configure(config)
        picam2.set_controls({"FrameRate": 30})
        picam2.start()

        time.sleep(1.0)

        frame_count = 0

        capture_times = []
        preprocess_times = []
        hailo_times = []
        postprocess_times = []
        encode_times = []
        total_times = []

        print("Streaming started. Press Ctrl+C to stop.")

        try:
            with InferVStreams(
                network_group,
                input_vstreams_params,
                output_vstreams_params,
            ) as infer_pipeline:
                with network_group.activate(network_group_params):

                    while True:
                        t0 = time.perf_counter()

                        frame_rgb = picam2.capture_array()
                        t1 = time.perf_counter()

                        lr, x = preprocess_frame(frame_rgb, lr_w, lr_h)
                        t2 = time.perf_counter()

                        result = infer_pipeline.infer({input_name: x})
                        t3 = time.perf_counter()

                        sr = postprocess(result[output_name])
                        display = make_display(lr, sr)
                        t4 = time.perf_counter()

                        ok, jpg = cv2.imencode(
                            ".jpg",
                            display,
                            [
                                int(cv2.IMWRITE_JPEG_QUALITY),
                                JPEG_QUALITY,
                            ],
                        )
                        t5 = time.perf_counter()

                        if ok:
                            with latest_lock:
                                latest_jpeg = jpg.tobytes()

                        frame_count += 1

                        capture_ms = (t1 - t0) * 1000.0
                        preprocess_ms = (t2 - t1) * 1000.0
                        hailo_ms = (t3 - t2) * 1000.0
                        postprocess_ms = (t4 - t3) * 1000.0
                        encode_ms = (t5 - t4) * 1000.0
                        total_ms = (t5 - t0) * 1000.0

                        capture_times.append(capture_ms)
                        preprocess_times.append(preprocess_ms)
                        hailo_times.append(hailo_ms)
                        postprocess_times.append(postprocess_ms)
                        encode_times.append(encode_ms)
                        total_times.append(total_ms)

                        if frame_count % PRINT_EVERY == 0:
                            fps = 1000.0 / np.mean(
                                total_times[-PRINT_EVERY:]
                            )
                            cpu = psutil.cpu_percent(interval=None)
                            ram = process.memory_info().rss / (1024 ** 2)

                            print(
                                f"FPS: {fps:.2f} | "
                                f"capture: {np.mean(capture_times[-PRINT_EVERY:]):.2f} ms | "
                                f"pre: {np.mean(preprocess_times[-PRINT_EVERY:]):.2f} ms | "
                                f"hailo: {np.mean(hailo_times[-PRINT_EVERY:]):.2f} ms | "
                                f"post: {np.mean(postprocess_times[-PRINT_EVERY:]):.2f} ms | "
                                f"jpeg: {np.mean(encode_times[-PRINT_EVERY:]):.2f} ms | "
                                f"total: {np.mean(total_times[-PRINT_EVERY:]):.2f} ms | "
                                f"CPU: {cpu:.1f}% | "
                                f"RAM: {ram:.1f} MB",
                                flush=True,
                            )

        except KeyboardInterrupt:
            print("\nStopping stream...")
            running = False

            if total_times:
                print("\nFinal average timing")
                print("--------------------")
                print(f"Frames:     {frame_count}")
                print(f"FPS:        {1000.0 / np.mean(total_times):.2f}")
                print(f"Capture:    {np.mean(capture_times):.2f} ms")
                print(f"Preprocess: {np.mean(preprocess_times):.2f} ms")
                print(f"Hailo:      {np.mean(hailo_times):.2f} ms")
                print(f"Postprocess:{np.mean(postprocess_times):.2f} ms")
                print(f"JPEG:       {np.mean(encode_times):.2f} ms")
                print(f"Total:      {np.mean(total_times):.2f} ms")

        finally:
            picam2.stop()


if __name__ == "__main__":
    main()
