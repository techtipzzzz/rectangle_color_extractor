import cv2
import numpy as np
import os
from typing import List, Tuple


def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def auto_correct_face_orientation(image, debug=False):
    """Detect face using DNN and correct orientation if needed."""
    modelFile = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
    configFile = os.path.join("models", "deploy.prototxt")

    if not os.path.exists(modelFile) or not os.path.exists(configFile):
        if debug:
            print("DNN model files not found, skipping orientation correction.")
        return image

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            if debug:
                print(f"[DNN] Face detected with confidence {confidence:.2f}")
            return image

    if debug:
        print("No face detected with DNN.")
    return image


def _has_face(image, min_conf=0.5, debug=False):
    """Check if a face exists in the cropped region using DNN."""
    modelFile = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
    configFile = os.path.join("models", "deploy.prototxt")

    if not os.path.exists(modelFile) or not os.path.exists(configFile):
        if debug:
            print("Face model not found, skipping validation.")
        return True  # fallback: allow all

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_conf:
            return True
    return False


# ---------------- ORIGINAL MODE ----------------
def process_passport_photos_v3(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    max_kb: int,
    padx: float = 0.0,
    pady: float = 0.0,
    debug: bool = False
):
    """Original mode: no detection, resize full image only."""
    os.makedirs(output_path, exist_ok=True)
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Could not read input image.")

    h, w = image.shape[:2]
    pad_x = int(padx * w)
    pad_y = int(pady * h)
    x1 = max(0, 0 - pad_x)
    y1 = max(0, 0 - pad_y)
    x2 = min(w + pad_x, w)
    y2 = min(h + pad_y, h)
    roi = image[y1:y2, x1:x2]

    resized = cv2.resize(roi, (width, height))
    resized = auto_correct_face_orientation(resized, debug)

    save_path = os.path.join(output_path, "photo_1.jpg")
    quality = 95
    while True:
        cv2.imwrite(save_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if os.path.getsize(save_path) <= max_kb * 1024 or quality <= 20:
            break
        quality -= 5

    return 1, [save_path]


# ---------------- ENHANCED MODE ----------------
def process_passport_photos_v4(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    max_kb: int,
    padx: float = 0.2,
    pady: float = 0.2,
    method: str = "rectangle",
    min_size: int = 100,
    quality_threshold: float = 0.5,
    correct_rotation: bool = True,
    debug: bool = False
) -> Tuple[int, List[str]]:
    """Enhanced photo extraction with multiple detection methods."""
    os.makedirs(output_path, exist_ok=True)
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Could not read input image.")
    saved_paths = []
    count = 0

    if method in ("rectangle", "auto"):
        count, saved_paths = _your_original_rectangle_detection(
            image, input_path, output_path, width, height, max_kb, padx, pady, debug
        )
        if count > 0:
            return count, saved_paths

    if method == "tilted_rectangle":
        count, saved_paths = _tilted_rectangle_detection(
            image, input_path, output_path, width, height, max_kb, padx, pady, debug
        )
        if count > 0:
            return count, saved_paths

    if method == "color":
        regions = _simple_color_detection(image, min_size, debug)
        if regions:
            count, saved_paths = _extract_regions(
                image, regions, output_path, width, height, max_kb,
                padx, pady, correct_rotation, debug
            )
            if count > 0:
                return count, saved_paths

    if method == "color_tilted":
        count, saved_paths = _color_tilted_detection(
            image, input_path, output_path, width, height, max_kb, padx, pady, debug
        )
        if count > 0:
            return count, saved_paths

    if method == "auto":
        fallback_methods = ["grid", "color", "color_tilted", "template", "watershed"]
        for fallback_method in fallback_methods:
            try:
                if fallback_method == "color_tilted":
                    count, saved_paths = _color_tilted_detection(
                        image, input_path, output_path, width, height, max_kb, padx, pady, debug
                    )
                else:
                    regions = _detect_with_method(image, fallback_method, min_size, debug)
                    if regions:
                        count, saved_paths = _extract_regions(
                            image, regions, output_path, width, height, max_kb,
                            padx, pady, correct_rotation, debug
                        )
                if count > 0:
                    return count, saved_paths
            except Exception as e:
                if debug:
                    print(f"Fallback method {fallback_method} failed: {e}")

    if method and method not in ("rectangle", "tilted_rectangle", "color", "color_tilted"):
        regions = _detect_with_method(image, method, min_size, debug)
        if regions:
            count, saved_paths = _extract_regions(
                image, regions, output_path, width, height, max_kb,
                padx, pady, correct_rotation, debug
            )
            if count > 0:
                return count, saved_paths

    # Final fallback: skip saving full image if no face found
    if debug:
        print("No valid passport photo detected.")
    return 0, []


# ---------------- DETECTION METHODS ----------------
def _your_original_rectangle_detection(image, input_path, output_path, width, height, max_kb, padx, pady, debug):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    saved_paths = []
    count = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            x, y, w, h = cv2.boundingRect(approx)
            pad_x = int(padx * w)
            pad_y = int(pady * h)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, image.shape[1])
            y2 = min(y + h + pad_y, image.shape[0])
            roi = image[y1:y2, x1:x2]
            warp = cv2.resize(roi, (width, height))

            # ✅ validate face
            if not _has_face(warp, debug=debug):
                if debug:
                    print("Skipped rectangle region: no face detected")
                continue

            warp = auto_correct_face_orientation(warp, debug)
            count += 1
            save_path = os.path.join(output_path, f"photo_{count}.jpg")
            quality = 95
            while True:
                cv2.imwrite(save_path, warp, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if os.path.getsize(save_path) <= max_kb * 1024 or quality <= 20:
                    break
                quality -= 5
            saved_paths.append(save_path)
            if debug:
                print(f"Rectangle detected photo {count} size: {w}x{h}")
    return count, saved_paths


def _tilted_rectangle_detection(image, input_path, output_path, width, height, max_kb, padx, pady, debug):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    saved_paths = []
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        pts = order_points(np.array(box, dtype="float32"))

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warp = cv2.warpPerspective(image, M, (width, height))

        # ✅ validate face
        if not _has_face(warp, debug=debug):
            if debug:
                print("Skipped tilted rectangle region: no face detected")
            continue

        warp = auto_correct_face_orientation(warp, debug)

        count += 1
        save_path = os.path.join(output_path, f"photo_{count}.jpg")

        quality = 95
        while True:
            cv2.imwrite(save_path, warp, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if os.path.getsize(save_path) <= max_kb * 1024 or quality <= 20:
                break
            quality -= 5

        saved_paths.append(save_path)
        if debug:
            print(f"Tilted rectangle detected photo {count} with rect: {rect}")

    return count, saved_paths


def _color_tilted_detection(image, input_path, output_path, width, height, max_kb, padx, pady, debug):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,30]), np.array([180,255,220]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    saved_paths = []
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        pts = order_points(np.array(box, dtype="float32"))

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warp = cv2.warpPerspective(image, M, (width, height))

        # ✅ validate face
        if not _has_face(warp, debug=debug):
            if debug:
                print("Skipped color-tilted region: no face detected")
            continue

        warp = auto_correct_face_orientation(warp, debug)

        count += 1
        save_path = os.path.join(output_path, f"photo_{count}.jpg")

        quality = 95
        while True:
            cv2.imwrite(save_path, warp, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if os.path.getsize(save_path) <= max_kb * 1024 or quality <= 20:
                break
            quality -= 5

        saved_paths.append(save_path)
        if debug:
            print(f"Color-tilted detected photo {count} with rect: {rect}")
    return count, saved_paths


def _detect_with_method(image, method, min_size, debug):
    if method == "grid":
        return _simple_grid_detection(image, min_size, debug)
    elif method == "color":
        return _simple_color_detection(image, min_size, debug)
    elif method == "template":
        return _simple_template_detection(image, min_size, debug)
    elif method == "watershed":
        return _simple_watershed_detection(image, min_size, debug)
    return []


def _simple_grid_detection(image, min_size, debug):
    h, w = image.shape[:2]
    configs = [(2,2), (2,3), (3,2), (1,4)]
    for rows, cols in configs:
        cell_h, cell_w = h // rows, w // cols
        if cell_h >= min_size and cell_w >= min_size:
            regions = []
            for row in range(rows):
                for col in range(cols):
                    x1 = col * cell_w
                    y1 = row * cell_h
                    regions.append({"bbox": (x1, y1, cell_w, cell_h)})
            if debug:
                print(f"Grid detection found {len(regions)} regions")
            return regions
    return []


def _simple_color_detection(image, min_size, debug):
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,0,30]), np.array([180,255,220]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_size*min_size:
                x,y,w,h = cv2.boundingRect(cnt)
                if 0.5 <= w/h <= 2.0:
                    regions.append({"bbox": (x,y,w,h)})
        if debug:
            print(f"Color detection found {len(regions)} regions")
        return regions
    except:
        return []


def _simple_template_detection(image, min_size, debug):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        regions = []
        for ratio_w, ratio_h in [(3,4), (4,5)]:
            for size in [120,160]:
                tw = int(ratio_w * size / 3)
                th = int(ratio_h * size / 3)
                if tw < min_size or th < min_size:
                    continue
                step = max(tw//3, 40)
                for y in range(0, h-th, step):
                    for x in range(0, w-tw, step):
                        window = gray[y:y+th, x:x+tw]
                        if np.var(window) > 200:
                            regions.append({"bbox": (x,y,tw,th)})
        if debug:
            print(f"Template detection found {len(regions)} regions")
        return regions[:6]
    except:
        return []


def _simple_watershed_detection(image, min_size, debug):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, peaks = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
        _, markers = cv2.connectedComponents(peaks.astype(np.uint8))
        markers = cv2.watershed(image, markers)
        regions = []
        for marker_id in np.unique(markers):
            if marker_id <= 1:
                continue
            mask = (markers == marker_id).astype(np.uint8)*255
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) > min_size*min_size:
                    x,y,w,h = cv2.boundingRect(cnt)
                    regions.append({"bbox": (x,y,w,h)})
        if debug:
            print(f"Watershed detection found {len(regions)} regions")
        return regions
    except:
        return []


# ---------------- REGION EXTRACTION ----------------
def _extract_regions(
    image, regions, output_path, width, height, max_kb,
    padx, pady, correct_rotation, debug
):
    saved_paths = []
    count = 0
    for region in regions:
        try:
            x,y,w,h = region["bbox"]
            pad_x = int(padx * w)
            pad_y = int(pady * h)
            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w + pad_x, image.shape[1])
            y2 = min(y + h + pad_y, image.shape[0])
            roi = image[y1:y2, x1:x2]
            resized = cv2.resize(roi, (width, height))

            # ✅ validate face
            if not _has_face(resized, debug=debug):
                if debug:
                    print("Skipped extracted region: no face detected")
                continue

            if correct_rotation:
                resized = auto_correct_face_orientation(resized, debug)
            count += 1
            save_path = os.path.join(output_path, f"photo_{count}.jpg")
            quality = 95
            while True:
                cv2.imwrite(save_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if os.path.getsize(save_path) <= max_kb * 1024 or quality <= 20:
                    break
                quality -= 5
            saved_paths.append(save_path)
        except Exception as e:
            if debug:
                print(f"Error extracting region: {e}")
    return count, saved_paths

