from __future__ import annotations

import argparse
import ctypes
import os
import statistics
from pathlib import Path

import numpy as np

ONI_API_VERSION = 2002
ONI_STATUS_OK = 0
ONI_SENSOR_DEPTH = 3
ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION = 5
ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR = 1


class OniVideoMode(ctypes.Structure):
    _fields_ = [
        ("pixelFormat", ctypes.c_int),
        ("resolutionX", ctypes.c_int),
        ("resolutionY", ctypes.c_int),
        ("fps", ctypes.c_int),
    ]


class OniFrame(ctypes.Structure):
    _fields_ = [
        ("dataSize", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("sensorType", ctypes.c_int),
        ("timestamp", ctypes.c_uint64),
        ("frameIndex", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("videoMode", OniVideoMode),
        ("croppingEnabled", ctypes.c_int),
        ("cropOriginX", ctypes.c_int),
        ("cropOriginY", ctypes.c_int),
        ("stride", ctypes.c_int),
    ]


OniDeviceHandle = ctypes.c_void_p
OniStreamHandle = ctypes.c_void_p
OniFramePtr = ctypes.POINTER(OniFrame)


class OpenNI:
    def __init__(self, redist: Path) -> None:
        dll_path = find_openni_dll(redist)
        dll_dir = dll_path.parent
        self.dll_path = dll_path

        self._dll_dir_cookie = None
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_cookie = os.add_dll_directory(str(dll_dir))

        self.dll = ctypes.WinDLL(str(dll_path))
        self._configure_functions()

    def _configure_functions(self) -> None:
        self.dll.oniInitialize.restype = ctypes.c_int
        self.dll.oniInitialize.argtypes = [ctypes.c_int]

        self.dll.oniShutdown.restype = None
        self.dll.oniShutdown.argtypes = []

        self.dll.oniDeviceOpen.restype = ctypes.c_int
        self.dll.oniDeviceOpen.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(OniDeviceHandle),
        ]

        self.dll.oniDeviceClose.restype = ctypes.c_int
        self.dll.oniDeviceClose.argtypes = [OniDeviceHandle]

        self.dll.oniDeviceSetProperty.restype = ctypes.c_int
        self.dll.oniDeviceSetProperty.argtypes = [
            OniDeviceHandle,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        self.dll.oniDeviceIsImageRegistrationModeSupported.restype = ctypes.c_int
        self.dll.oniDeviceIsImageRegistrationModeSupported.argtypes = [
            OniDeviceHandle,
            ctypes.c_int,
        ]

        self.dll.oniDeviceCreateStream.restype = ctypes.c_int
        self.dll.oniDeviceCreateStream.argtypes = [
            OniDeviceHandle,
            ctypes.c_int,
            ctypes.POINTER(OniStreamHandle),
        ]

        self.dll.oniStreamStart.restype = ctypes.c_int
        self.dll.oniStreamStart.argtypes = [OniStreamHandle]

        self.dll.oniStreamStop.restype = None
        self.dll.oniStreamStop.argtypes = [OniStreamHandle]

        self.dll.oniStreamDestroy.restype = None
        self.dll.oniStreamDestroy.argtypes = [OniStreamHandle]

        self.dll.oniStreamReadFrame.restype = ctypes.c_int
        self.dll.oniStreamReadFrame.argtypes = [
            OniStreamHandle,
            ctypes.POINTER(OniFramePtr),
        ]

        self.dll.oniFrameRelease.restype = None
        self.dll.oniFrameRelease.argtypes = [OniFramePtr]

    def close_dll_directory(self) -> None:
        if self._dll_dir_cookie is not None:
            self._dll_dir_cookie.close()
            self._dll_dir_cookie = None


def require_ok(status: int, action: str) -> None:
    if status != ONI_STATUS_OK:
        raise RuntimeError(f"{action} failed, OniStatus={status}")


def try_enable_registration(openni: OpenNI, device: OniDeviceHandle) -> None:
    is_supported = bool(
        openni.dll.oniDeviceIsImageRegistrationModeSupported(
            device,
            ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR,
        ),
    )

    print(f"registration_depth_to_color_supported={is_supported}")

    if not is_supported:
        print("warning: OpenNI reports depth-to-color registration is not supported.")
        return

    registration_mode = ctypes.c_int(ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    status = openni.dll.oniDeviceSetProperty(
        device,
        ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION,
        ctypes.byref(registration_mode),
        ctypes.sizeof(registration_mode),
    )

    if status == ONI_STATUS_OK:
        print("registration_depth_to_color=enabled")
        return

    print(
        "warning: failed to enable depth-to-color registration; "
        f"OniStatus={status}. Continue reading raw depth.",
    )


def find_openni_dll(path: Path) -> Path:
    if path.is_file() and path.name.lower() == "openni2.dll":
        return path.resolve()

    if path.is_dir():
        direct_dll = path / "OpenNI2.dll"

        if direct_dll.exists():
            return direct_dll.resolve()

        matches = sorted(path.rglob("OpenNI2.dll"), key=score_openni_dll)

        if matches:
            return matches[0].resolve()

    fallback_root = Path("OpenNI_2.3.0.86")

    if fallback_root.exists() and fallback_root.resolve() != path.resolve():
        matches = sorted(fallback_root.rglob("OpenNI2.dll"), key=score_openni_dll)

        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        "OpenNI2.dll not found. Pass the OpenNI root directory, "
        "the Redist directory, or the exact OpenNI2.dll path.",
    )


def score_openni_dll(path: Path) -> tuple[int, str]:
    text = str(path).replace("\\", "/").lower()
    score = 100

    if "win64-release" in text:
        score -= 50

    if "redist" in text:
        score -= 20

    if "niviewer" in text:
        score -= 18

    if "/tools/" in text:
        score -= 12

    if "/bin/" in text:
        score -= 8

    if "samples" in text:
        score += 30

    if "win32-release" in text:
        score += 40

    return score, text


def list_openni_dlls(path: Path) -> list[Path]:
    roots = [path]
    fallback_root = Path("OpenNI_2.3.0.86")

    if fallback_root.exists() and fallback_root.resolve() != path.resolve():
        roots.append(fallback_root)

    matches: dict[Path, None] = {}

    for root in roots:
        if root.is_file() and root.name.lower() == "openni2.dll":
            matches[root.resolve()] = None
        elif root.is_dir():
            for match in root.rglob("OpenNI2.dll"):
                matches[match.resolve()] = None

    return sorted(matches.keys(), key=score_openni_dll)


def get_nonzero_median(depth: np.ndarray, x: int, y: int, radius: int) -> int:
    height, width = depth.shape
    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)
    patch = depth[y1:y2, x1:x2]
    values = patch[patch > 0]

    if values.size == 0:
        return 0

    return int(np.median(values))


def resolve_sample_point(
    args: argparse.Namespace,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, str]:
    if args.norm_x is not None or args.norm_y is not None:
        if args.norm_x is None or args.norm_y is None:
            raise ValueError("--norm-x and --norm-y must be used together.")

        x = round(float(args.norm_x) * (frame_width - 1))
        y = round(float(args.norm_y) * (frame_height - 1))
        label = f"norm=({args.norm_x:.3f},{args.norm_y:.3f})"
    elif args.x is not None or args.y is not None:
        if args.x is None or args.y is None:
            raise ValueError("--x and --y must be used together.")

        x = int(args.x)
        y = int(args.y)
        label = f"pixel=({x},{y})"
    else:
        x = frame_width // 2
        y = frame_height // 2
        label = "center"

    x = min(max(x, 0), frame_width - 1)
    y = min(max(y, 0), frame_height - 1)

    return x, y, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read ORBBEC/OpenNI depth frames and print center depth.",
    )
    parser.add_argument(
        "--openni-redist",
        default="OpenNI_2.3.0.86",
        help="Path to OpenNI root, Redist directory, or OpenNI2.dll.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of depth frames to sample.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Median filter radius around center point.",
    )
    parser.add_argument(
        "--x",
        type=int,
        default=None,
        help="Pixel x to sample. Use with --y.",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=None,
        help="Pixel y to sample. Use with --x.",
    )
    parser.add_argument(
        "--norm-x",
        type=float,
        default=None,
        help="Normalized x in [0, 1], matching MediaPipe-style coordinates. Use with --norm-y.",
    )
    parser.add_argument(
        "--norm-y",
        type=float,
        default=None,
        help="Normalized y in [0, 1], matching MediaPipe-style coordinates. Use with --norm-x.",
    )
    parser.add_argument(
        "--enable-registration",
        action="store_true",
        help="Try to enable OpenNI depth-to-color registration before reading depth.",
    )
    parser.add_argument(
        "--list-dlls",
        action="store_true",
        help="List discovered OpenNI2.dll candidates and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    redist = Path(args.openni_redist).resolve()

    if args.list_dlls:
        candidates = list_openni_dlls(redist)

        if not candidates:
            print("No OpenNI2.dll candidates found.")
            return

        for index, candidate in enumerate(candidates):
            print(f"{index}: {candidate}")

        return

    openni = OpenNI(redist)
    device = OniDeviceHandle()
    stream = OniStreamHandle()
    stream_started = False

    try:
        print(f"openni_dll={openni.dll_path}")
        require_ok(openni.dll.oniInitialize(ONI_API_VERSION), "oniInitialize")
        require_ok(openni.dll.oniDeviceOpen(None, ctypes.byref(device)), "oniDeviceOpen")

        if args.enable_registration:
            try_enable_registration(openni, device)

        require_ok(
            openni.dll.oniDeviceCreateStream(
                device,
                ONI_SENSOR_DEPTH,
                ctypes.byref(stream),
            ),
            "oniDeviceCreateStream(depth)",
        )
        require_ok(openni.dll.oniStreamStart(stream), "oniStreamStart(depth)")
        stream_started = True

        samples: list[int] = []
        frame_width = 0
        frame_height = 0

        for index in range(args.frames):
            frame_ptr = OniFramePtr()
            require_ok(
                openni.dll.oniStreamReadFrame(stream, ctypes.byref(frame_ptr)),
                "oniStreamReadFrame(depth)",
            )

            try:
                frame = frame_ptr.contents
                frame_width = frame.width
                frame_height = frame.height
                byte_count = frame.dataSize
                raw = (ctypes.c_uint16 * (byte_count // 2)).from_address(frame.data)
                depth = np.ctypeslib.as_array(raw).reshape(frame_height, frame_width)
                sample_x, sample_y, sample_label = resolve_sample_point(
                    args,
                    frame_width,
                    frame_height,
                )
                sample_depth = int(depth[sample_y, sample_x])
                median_depth = get_nonzero_median(
                    depth,
                    sample_x,
                    sample_y,
                    args.radius,
                )
            finally:
                openni.dll.oniFrameRelease(frame_ptr)

            if median_depth > 0:
                samples.append(median_depth)

            if index % 10 == 0:
                print(
                    f"frame={index:03d} "
                    f"size={frame_width}x{frame_height} "
                    f"{sample_label}=({sample_x},{sample_y}) "
                    f"depth={sample_depth}mm "
                    f"median_r{args.radius}={median_depth}mm"
                )

        if samples:
            print("---- summary ----")
            print(f"valid_samples={len(samples)}/{args.frames}")
            print(f"median={int(statistics.median(samples))}mm")
            print(f"mean={statistics.mean(samples):.1f}mm")
            print(f"min={min(samples)}mm")
            print(f"max={max(samples)}mm")
            print(f"stdev={statistics.pstdev(samples):.1f}mm")
        else:
            print("No valid non-zero depth samples at center point.")
    finally:
        if stream_started:
            openni.dll.oniStreamStop(stream)

        if stream:
            openni.dll.oniStreamDestroy(stream)

        if device:
            openni.dll.oniDeviceClose(device)

        openni.dll.oniShutdown()
        openni.close_dll_directory()


if __name__ == "__main__":
    main()
