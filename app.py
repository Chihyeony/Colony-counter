import csv
import io
import math
from collections import Counter

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# ⚙️ 상수
# ---------------------------------------------------------
MAX_SIDE = 1500            # 처리용 최대 이미지 변 길이 (px)
MAX_FILES = 10
TNTC_LIMIT = 300           # 일반적인 계수 가능 범위 30~300 CFU
TFTC_LIMIT = 30
DEFAULT_CIRCULARITY = 0.35
MOVE_STEP = 10             # 영역 위치 미세 조정 단위 (처리 이미지 px)
MARKER_MAX_COVERAGE = 0.25 # 배지의 25% 이상이 '네임펜'이면 배지 색으로 판단하고 제거를 끔
ALL_IMAGES = -1

# UI 기본 설정
st.set_page_config(
    page_title="Auto Colony Counter",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 커스텀 CSS 스타일링 ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stButton"] button {
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

if 'offsets' not in st.session_state:
    st.session_state.offsets = {}   # 이미지별 영역 위치 보정값 {file_key: (x, y)}


# ---------------------------------------------------------
# 🧩 기본 유틸
# ---------------------------------------------------------
def decode_and_normalize(file_bytes):
    """이미지를 읽고 리사이즈/세로 방향으로 맞춘다. 읽기 실패 시 None."""
    img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def disk_area(diameter):
    return math.pi * (diameter / 2) ** 2


def detect_dish(gray):
    """
    배지 경계를 찾는다.
    - 밝은 영역/어두운 영역 양쪽 후보를 모두 보고 '크고 둥근' 덩어리를 고른다
      (밝은 배경에서 찍어 배지가 더 어두운 경우도 대응).
    - 가장자리 콜로니/글씨로 생긴 홈의 영향을 줄이려 볼록 껍질(convex hull)에 타원을 맞춘다.
    반환: (ellipse, found)
    """
    h, w = gray.shape
    img_area = float(h * w)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_hull, best_score = None, 0.0
    for cand in (th, cv2.bitwise_not(th)):
        contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.10 * img_area:
                continue
            hull = cv2.convexHull(c)
            if len(hull) < 5:
                continue
            _, radius = cv2.minEnclosingCircle(hull)
            if radius <= 0:
                continue
            roundness = cv2.contourArea(hull) / (math.pi * radius * radius)
            _, _, bw, bh = cv2.boundingRect(c)
            if bw >= w - 2 and bh >= h - 2:   # 이미지 전체를 감싸는 덩어리 = 배경일 가능성
                roundness *= 0.3
            score = roundness * math.sqrt(area / img_area)
            if score > best_score:
                best_score, best_hull = score, hull

    if best_hull is None:
        r = min(h, w) * 0.45
        return ((w / 2, h / 2), (2 * r, 2 * r), 0.0), False
    return cv2.fitEllipse(best_hull), True


def make_dish_mask(shape, ellipse, edge_crop, offset):
    (cx, cy), (a1, a2), angle = ellipse
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, ((cx + offset[0], cy + offset[1]), (a1 * edge_crop, a2 * edge_crop), angle), 255, -1)
    return mask


def build_marker_mask(bgr, dish_mask, remove_dark, remove_color, max_colony_area):
    """
    네임펜 영역 추출.
    - 임계값을 배지 자체의 밝기/채도 기준으로 상대화 → 혈액한천 등 유색 배지 전체가 잡히는 것 방지
    - 둥글고 콜로니 크기인 덩어리는 펜이 아니라 (색소) 콜로니로 보고 제외
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)
    inside = dish_mask > 0
    v_med = float(np.median(v_ch[inside]))
    s_med = float(np.median(s_ch[inside]))

    raw = np.zeros_like(v_ch)
    if remove_dark:
        raw |= cv2.inRange(v_ch, 0, int(min(50, 0.35 * v_med)))
    if remove_color:
        raw |= cv2.inRange(s_ch, int(min(250, max(110, s_med + 60))), 255)
    raw = cv2.bitwise_and(raw, dish_mask)

    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    colony_like = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0 or area <= 0 or area > max_colony_area:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(c)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [c - [x, y]], -1, 255, -1)
        fill_ratio = np.count_nonzero(raw[y:y + h, x:x + w] & filled) / max(1, np.count_nonzero(filled))
        # 속이 꽉 찬 원 = 콜로니, 'O'/'0' 같은 고리 모양 글자는 fill_ratio가 낮음
        if circularity >= 0.7 and fill_ratio >= 0.85:
            colony_like.append(c)
    cv2.drawContours(raw, colony_like, -1, 0, -1)

    marker = cv2.dilate(raw, np.ones((5, 5), np.uint8), iterations=2)
    return cv2.bitwise_and(marker, dish_mask)


def estimate_background(gray, dish_mask, window_px):
    """콜로니보다 충분히 큰 창의 메디안 필터로 배지 배경(조명 불균일 포함)을 추정."""
    h, w = gray.shape
    f = 4  # 속도를 위해 1/4 축소 후 계산
    size = (max(1, w // f), max(1, h // f))
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    small_inside = cv2.resize(dish_mask, size, interpolation=cv2.INTER_NEAREST) > 0
    if np.any(small_inside):
        # 배지 바깥을 '가장 가까운 배지 안쪽 픽셀 값'으로 채움 → 가장자리에서도 배경 추정이 정확
        _, nn = cv2.distanceTransformWithLabels(
            np.where(small_inside, 0, 255).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(nn.max() + 1, np.uint8)
        lut[nn[small_inside]] = small[small_inside]
        small = lut[nn]
    k = int(window_px / f) | 1
    k = max(5, min(k, (min(small.shape) // 2) * 2 - 1))
    bg_small = cv2.medianBlur(small, k)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), k / 6)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def h_maxima(dist, h):
    """돌출 높이(prominence)가 h 이상인 봉우리 꼭대기만 남긴다 (형태학적 재구성)."""
    marker = np.maximum(dist - h, 0).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(2000):
        grown = np.minimum(cv2.dilate(marker, kernel), dist)
        if np.array_equal(grown, marker):
            break
        marker = grown
    dome = dist - marker
    return (dome >= h - 1e-3) & (dist > 0)


def find_seeds(binary, min_d_px):
    """콜로니 중심 후보(씨앗): 돌출 높이가 충분한 거리변환 봉우리만 (가장자리 요철로 인한 과분할 방지)."""
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.0)   # 테두리 노이즈로 생기는 가짜 봉우리 완화
    r_min = min_d_px / 2
    seeds = h_maxima(dist, max(1.5, 0.4 * r_min)) & (dist >= 0.5 * r_min) & (opened > 0)
    return opened, dist, seeds


def merge_weak_splits(markers, dist, ratio=0.85):
    """
    나뉜 두 영역 사이 경계(잘록한 곳)의 두께가 양쪽 봉우리의 ratio 이상이면
    실제로는 한 덩어리(길쭉한 콜로니/반사광)로 보고 다시 합친다.
    """
    n = int(markers.max()) + 1
    if n <= 3:
        return markers
    peaks = np.zeros(n, np.float32)
    pos = markers >= 2
    np.maximum.at(peaks, markers[pos], dist[pos])

    ys, xs = np.nonzero(markers == -1)
    if len(ys) == 0:
        return markers
    H, W = markers.shape
    neigh = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neigh.append(markers[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)])
    neigh = np.stack(neigh)
    lo = np.where(neigh >= 2, neigh, np.iinfo(np.int32).max).min(axis=0)
    hi = neigh.max(axis=0)
    ok = (lo >= 2) & (hi > lo) & (lo != np.iinfo(np.int32).max)

    saddle = {}
    for a, b, d in zip(lo[ok], hi[ok], dist[ys[ok], xs[ok]]):
        key = (int(a), int(b))
        if d > saddle.get(key, 0):
            saddle[key] = float(d)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), d in saddle.items():
        if d >= ratio * min(peaks[a], peaks[b]):
            parent[find(b)] = find(a)

    lut = np.array([find(i) for i in range(n)], np.int32)
    out = markers.copy()
    out[pos] = lut[markers[pos]]
    # 합쳐진 영역 사이의 경계선(-1)은 해당 영역으로 편입
    same = ok & (lut[np.clip(lo, 0, n - 1)] == lut[np.clip(hi, 0, n - 1)])
    out[ys[same], xs[same]] = lut[lo[same]]
    return out


def segment(prep):
    """거리 지형 기반 watershed로 콜로니 영역을 나눈다. 반환: (markers, opened_binary). 콜로니 라벨은 2 이상."""
    binary, dist, seeds = find_seeds(prep["binary"], prep["min_d_px"])
    _, markers = cv2.connectedComponents(seeds.astype(np.uint8))
    markers = (markers + 1) * seeds                # 씨앗=2.., 나머지=0(미정)

    # 밝기 경사가 아닌 '거리' 지형에서 분할 → 붙은 콜로니 사이 잘록한 곳에서 나뉨
    # (씨앗끼리만 경쟁시키고, 끝난 뒤 콜로니 영역(binary) 밖은 버림)
    # 콜로니 내부는 0~254, 배경은 255 → 콜로니 픽셀이 항상 자기 씨앗에서 먼저 채워짐
    scale = 200.0 / max(1.0, prep["max_d_px"] / 2)
    landscape = (254 - np.clip(dist * scale, 0, 254)).astype(np.uint8)
    landscape[binary == 0] = 255
    markers = cv2.watershed(cv2.cvtColor(landscape, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
    markers = merge_weak_splits(markers, dist)

    # 콜로니 영역 밖, 그리고 씨앗이 없는(너무 작은) 덩어리는 버림
    _, comp = cv2.connectedComponents(binary)
    seeded = np.unique(comp[seeds])
    markers[(binary == 0) | ~np.isin(comp, seeded)] = 0
    return markers, binary


# ---------------------------------------------------------
# 🧹 전처리 (배지 인식 → 네임펜 제거 → 배경 보정 → 콜로니 신호 이미지)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=40)
def prepare(file_bytes, offset, cfg):
    bgr = decode_and_normalize(file_bytes)
    if bgr is None:
        return {"error": "이미지를 읽을 수 없어요 (손상되었거나 지원하지 않는 형식)."}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    warnings = []

    # 1. 배지 인식 + px/mm 보정
    ellipse, found = detect_dish(gray)
    if not found:
        warnings.append("배지 경계를 찾지 못해 이미지 중앙 원을 사용했어요.")
    dish_d_px = max(ellipse[1])
    px_per_mm = dish_d_px / cfg["dish_mm"]
    min_d_px = max(2.0, cfg["min_d_mm"] * px_per_mm)
    max_d_px = max(min_d_px + 1, cfg["max_d_mm"] * px_per_mm)

    dish_mask = make_dish_mask(gray.shape, ellipse, cfg["edge_crop"], offset)
    inside = dish_mask > 0
    if not np.any(inside):
        return {"error": "분석 영역이 이미지 밖으로 벗어났어요. 위치 보정값을 초기화해 주세요."}

    # 2. 네임펜 제거
    exclusion = np.zeros_like(gray)
    if cfg["remove_dark"] or cfg["remove_color"]:
        marker = build_marker_mask(bgr, dish_mask, cfg["remove_dark"], cfg["remove_color"],
                                   disk_area(max_d_px))
        coverage = np.count_nonzero(marker) / np.count_nonzero(inside)
        if coverage > MARKER_MAX_COVERAGE:
            warnings.append(f"네임펜 후보가 배지의 {coverage:.0%}를 차지해 배지 색으로 판단, 이 이미지는 네임펜 제거를 껐어요.")
        else:
            exclusion = marker
    gray_clean = cv2.inpaint(gray, exclusion, 3, cv2.INPAINT_TELEA) if np.any(exclusion) else gray

    # 3. 배경(조명 불균일) 보정
    bg = estimate_background(gray_clean, dish_mask, max(3 * max_d_px, dish_d_px / 8))
    diff = gray_clean.astype(np.int16) - bg.astype(np.int16)

    valid = inside & (exclusion == 0)
    d_in = diff[valid] if np.any(valid) else diff[inside]
    med = float(np.median(d_in))
    sigma = max(1.0, 1.4826 * float(np.median(np.abs(d_in - med))))  # 강건한 노이즈 추정
    thr = max(float(cfg["min_contrast"]), 4.0 * sigma)

    # 4. 콜로니 밝기 방향(극성) 결정
    if cfg["polarity"] == "auto":
        n_bright = int(np.count_nonzero(d_in > thr))
        n_dark = int(np.count_nonzero(d_in < -thr))
        bright = n_bright >= n_dark
    else:
        bright = cfg["polarity"] == "bright"

    signal = np.clip(diff if bright else -diff, 0, 255).astype(np.uint8)
    signal[~inside] = 0
    signal[exclusion > 0] = 0
    binary = ((signal >= thr) & inside).astype(np.uint8) * 255

    return {
        "error": None,
        "bgr": bgr,
        "dish_mask": dish_mask,
        "exclusion": exclusion,
        "signal": signal,
        "binary": binary,
        "thr": thr,
        "bright": bright,
        "px_per_mm": px_per_mm,
        "min_d_px": min_d_px,
        "max_d_px": max_d_px,
        "excluded_ratio": np.count_nonzero(exclusion) / np.count_nonzero(inside),
        "warnings": warnings,
    }


# ---------------------------------------------------------
# 🧭 이미지 특성 기반 알고리즘 추천 (규칙 기반)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=40)
def recommend(file_bytes, offset, cfg):
    prep = prepare(file_bytes, offset, cfg)
    if prep["error"]:
        return "Blob", ""

    markers, opened = segment(prep)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    areas = stats[:, cv2.CC_STAT_AREA]
    # 덩어리마다 몇 개의 콜로니로 나뉘었는지
    pairs = np.unique(np.stack([labels[markers >= 2], markers[markers >= 2]]), axis=1)
    regions_per_comp = np.bincount(pairs[0], minlength=n_lbl)

    min_area = disk_area(prep["min_d_px"])
    max_area = disk_area(prep["max_d_px"])
    # 배경(0번), 노이즈(너무 작음), 반사광(나뉘지 않았는데 최대 크기 초과) 제외
    keep = (np.arange(n_lbl) > 0) & (areas >= min_area) & ~((regions_per_comp <= 1) & (areas > max_area))
    if not np.any(keep):
        return "Blob", "뚜렷한 콜로니 덩어리가 없음"

    n = int(keep.sum())
    merged = keep & (regions_per_comp >= 2)        # 한 덩어리가 2개 이상으로 나뉨 = 붙은 콜로니
    merged_frac = areas[merged].sum() / areas[keep].sum()
    if n > 100:
        return "Watershed", f"덩어리 {n}개 (>100, 밀집)"
    if merged_frac > 0.10:
        return "Watershed", f"붙은 콜로니 면적 {merged_frac:.0%} (>10%)"
    return "Blob", f"덩어리 {n}개, 붙은 콜로니 면적 {merged_frac:.0%}"


# ---------------------------------------------------------
# 🖼️ 결과 시각화 공통
# ---------------------------------------------------------
def base_overlay(prep, bg_opacity):
    bgr = prep["bgr"]
    dish = prep["dish_mask"]
    dark = (bgr.astype(np.float32) * bg_opacity).astype(np.uint8)
    img = np.where(dish[..., None] > 0, bgr, dark)

    # 🟡 제외된 네임펜 영역 (노란색 반투명)
    ex = prep["exclusion"] > 0
    if np.any(ex):
        yellow = np.array([0, 215, 255], np.float32)
        img[ex] = (0.4 * yellow + 0.6 * img[ex].astype(np.float32)).astype(np.uint8)

    contours, _ = cv2.findContours(dish, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    return img


# ---------------------------------------------------------
# 💧 1. Watershed 알고리즘
# ---------------------------------------------------------
def count_colonies_watershed(prep, bg_opacity):
    markers, _ = segment(prep)
    labels, counts = np.unique(markers, return_counts=True)
    is_colony = labels >= 2
    min_area, max_area = disk_area(prep["min_d_px"]), disk_area(prep["max_d_px"])
    valid_labels = labels[is_colony & (counts >= min_area) & (counts <= max_area)]
    oversized_labels = labels[is_colony & (counts > max_area)]

    img = base_overlay(prep, bg_opacity)
    for lbls, color in ((oversized_labels, (0, 0, 255)), (valid_labels, (0, 255, 0))):
        if len(lbls) == 0:
            continue
        m = np.isin(markers, lbls).astype(np.uint8) * 255
        c, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, c, -1, color, 2)
    return len(valid_labels), len(oversized_labels), img


# ---------------------------------------------------------
# 🔍 2. Blob Detection 알고리즘
# ---------------------------------------------------------
def count_colonies_blob(prep, min_circularity, bg_opacity):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = float(prep["thr"])       # 배경 보정된 신호 기준 → 배지 밝기와 무관
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.minDistBetweenBlobs = max(2.0, 0.7 * prep["min_d_px"])   # 맞닿은 콜로니도 각각 인식

    params.filterByArea = True
    params.minArea = max(1.0, disk_area(prep["min_d_px"]))
    params.maxArea = disk_area(prep["max_d_px"])

    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(prep["signal"])

    img = base_overlay(prep, bg_opacity)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        cv2.circle(img, (x, y), r + 2, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    return len(keypoints), 0, img


@st.cache_data(show_spinner=False, max_entries=40)
def run_count(file_bytes, offset, cfg, algo, min_circularity, bg_opacity):
    prep = prepare(file_bytes, offset, cfg)
    if algo == "Watershed":
        count, oversized, img = count_colonies_watershed(prep, bg_opacity)
    else:
        count, oversized, img = count_colonies_blob(prep, min_circularity, bg_opacity)
    return {
        "count": count,
        "oversized": oversized,
        "original_rgb": cv2.cvtColor(prep["bgr"], cv2.COLOR_BGR2RGB),
        "result_rgb": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    }


# ---------------------------------------------------------
# 3. 웹 UI 구성
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧫 Auto Colony Counter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">스마트 분석이 탑재된 콜로니 카운터 MADE BY 치현</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    f"콜로니 이미지를 드래그 앤 드롭 하세요 (.jpg, .png) - 최대 {MAX_FILES}장",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
uploaded_files = uploaded_files or []
if len(uploaded_files) > MAX_FILES:
    st.warning(f"⚠️ 최대 {MAX_FILES}개의 이미지만 분석합니다. 처음 {MAX_FILES}개만 처리됩니다.")
    uploaded_files = uploaded_files[:MAX_FILES]
file_keys = [f"{f.name}::{f.size}" for f in uploaded_files]

with st.sidebar:
    st.header("⚙️ 분석 방식 및 파라미터")

    selected_method = st.radio(
        "🧪 분석 알고리즘 선택",
        options=[
            "🌟 자동 선택 (이미지 특성 분석 후 적용)",
            "Watershed (콜로니 수가 많고 다닥다닥 붙었을 때)",
            "Blob (콜로니가 서로 떨어져 있을 때)"
        ],
        index=0
    )
    is_auto = "자동" in selected_method
    unify_batch = False
    if is_auto:
        unify_batch = st.checkbox(
            "배치 전체에 같은 알고리즘 적용 (다수결)", value=True,
            help="같은 실험의 플레이트끼리 결과를 비교할 수 있도록 모든 이미지에 같은 방식을 씁니다."
        )
    st.markdown("---")

    with st.expander("🔬 콜로니 판정 기준", expanded=True):
        polarity_label = st.radio(
            "콜로니 밝기",
            ["자동 판별", "배경보다 밝음", "배경보다 어두움"],
            horizontal=True,
            help="투과광 촬영처럼 콜로니가 배지보다 어둡게 보이면 '배경보다 어두움'을 선택하세요."
        )
        dish_mm = st.number_input("배지 지름 (mm)", 30.0, 200.0, 90.0, 5.0,
                                  help="픽셀↔mm 환산 기준. 일반 페트리 디쉬는 90mm.")
        min_d_mm = st.slider("최소 콜로니 지름 (mm)", 0.1, 5.0, 0.5, 0.1)
        max_d_mm = st.slider("최대 콜로니 지름 (mm)", 1.0, 30.0, 10.0, 0.5,
                             help="이보다 큰 덩어리(반사광, 얼룩 등)는 세지 않고 🔴빨간색으로 표시합니다.")
        if max_d_mm <= min_d_mm:
            st.warning("최대 지름이 최소 지름보다 커야 해요. 최소+0.5mm로 맞춥니다.")
            max_d_mm = min_d_mm + 0.5
        min_contrast = st.slider("최소 대비 (밝기 차)", 3, 80, 12, 1,
                                 help="배경 대비 이 값 이상 차이 나야 콜로니로 판정합니다. "
                                      "이미지 노이즈가 크면 자동으로 더 높아집니다.")

    with st.expander("🧹 특수 노이즈 제거", expanded=False):
        st.markdown("<p style='font-size:0.85em; color:gray;'>네임펜 글씨 등으로 인식된 영역은 🟡노란색으로 처리됩니다. "
                    "둥근 콜로니 모양은 지우지 않습니다.</p>", unsafe_allow_html=True)
        remove_dark_val = st.checkbox("🖊️ 검은 네임펜 무시", value=True, help="배지보다 훨씬 어두운 글씨/그림자를 지웁니다.")
        remove_color_val = st.checkbox("🖍️ 유색 네임펜 무시", value=True,
                                       help="배지보다 채도가 훨씬 높은 영역을 지웁니다. 유색 배지/색소 콜로니에서 문제가 되면 끄세요.")

    with st.expander("✂️ 인식 영역 설정", expanded=False):
        edge_crop_val = st.slider("영역 크기 설정", 0.70, 1.00, 0.95, 0.01,
                                  help="배지 반지름 대비 분석 영역. 0.95 = 면적의 약 90% 분석.")
        bg_opacity_val = st.slider("배경 투명도", 0.0, 1.0, 0.15, 0.05)

        st.markdown("<div style='text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 10px;'>"
                    f"영역 위치 미세 조정 ({MOVE_STEP}px 단위)</div>", unsafe_allow_html=True)
        if uploaded_files:
            target_idx = st.selectbox(
                "조정할 이미지",
                options=[ALL_IMAGES] + list(range(len(uploaded_files))),
                format_func=lambda i: "모든 이미지" if i == ALL_IMAGES else uploaded_files[i].name,
            )
            targets = file_keys if target_idx == ALL_IMAGES else [file_keys[target_idx]]

            def shift(dx, dy):
                for k in targets:
                    ox, oy = st.session_state.offsets.get(k, (0, 0))
                    st.session_state.offsets[k] = (ox + dx, oy + dy)

            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col1:
                if st.button("↖", width="stretch"): shift(-MOVE_STEP, -MOVE_STEP)
                if st.button("⬅", width="stretch"): shift(-MOVE_STEP, 0)
                if st.button("↙", width="stretch"): shift(-MOVE_STEP, MOVE_STEP)
            with col2:
                if st.button("⬆", width="stretch"): shift(0, -MOVE_STEP)
                if st.button("초기화", width="stretch"):
                    for k in targets:
                        st.session_state.offsets[k] = (0, 0)
                if st.button("⬇", width="stretch"): shift(0, MOVE_STEP)
            with col3:
                if st.button("↗", width="stretch"): shift(MOVE_STEP, -MOVE_STEP)
                if st.button("➡", width="stretch"): shift(MOVE_STEP, 0)
                if st.button("↘", width="stretch"): shift(MOVE_STEP, MOVE_STEP)

            if target_idx != ALL_IMAGES:
                ox, oy = st.session_state.offsets.get(targets[0], (0, 0))
                st.caption(f"현재 보정값: x={ox}, y={oy}")
        else:
            st.caption("이미지를 올리면 이미지별로 위치를 조정할 수 있어요.")

    circularity_val = DEFAULT_CIRCULARITY
    if "Blob" in selected_method or is_auto:
        with st.expander("🔍 Blob 세부 설정", expanded=("Blob" in selected_method)):
            circularity_val = st.slider("Blob 최소 원형도", 0.1, 1.0, DEFAULT_CIRCULARITY, 0.05)


prep_cfg = {
    "edge_crop": edge_crop_val,
    "remove_dark": remove_dark_val,
    "remove_color": remove_color_val,
    "polarity": {"자동 판별": "auto", "배경보다 밝음": "bright", "배경보다 어두움": "dark"}[polarity_label],
    "dish_mm": dish_mm,
    "min_d_mm": min_d_mm,
    "max_d_mm": max_d_mm,
    "min_contrast": min_contrast,
}

if uploaded_files:
    st.markdown(f"### 📊 총 {len(uploaded_files)}개의 이미지 분석 결과")
    st.markdown("---")

    # 1단계: 전처리 + 알고리즘 결정
    jobs = []
    with st.spinner("이미지 전처리 중..."):
        for f, key in zip(uploaded_files, file_keys):
            file_bytes = f.getvalue()
            offset = st.session_state.offsets.get(key, (0, 0))
            prep = prepare(file_bytes, offset, prep_cfg)
            job = {"file": f, "bytes": file_bytes, "offset": offset, "prep": prep, "algo": None, "reason": ""}
            if not prep["error"]:
                if is_auto:
                    job["algo"], job["reason"] = recommend(file_bytes, offset, prep_cfg)
                elif "Watershed" in selected_method:
                    job["algo"] = "Watershed"
                else:
                    job["algo"] = "Blob"
            jobs.append(job)

    if is_auto and unify_batch:
        votes = Counter(j["algo"] for j in jobs if j["algo"])
        if votes:
            # 동률이면 붙은 콜로니를 분리할 수 있는 Watershed 쪽으로
            majority = max(votes, key=lambda a: (votes[a], a == "Watershed"))
            for j in jobs:
                if j["algo"]:
                    if j["algo"] != majority:
                        j["reason"] = f"다수결로 {majority} 적용 (개별 추천: {j['algo']} — {j['reason']})"
                    j["algo"] = majority

    # 2단계: 카운팅 및 표시
    rows = []
    grid_cols = st.columns(3)
    for idx, job in enumerate(jobs):
        f, prep = job["file"], job["prep"]
        with grid_cols[idx % 3]:
            st.markdown(f"##### 🧪 {f.name}")
            if prep["error"]:
                st.error(prep["error"])
                rows.append({"파일명": f.name, "카운트": "", "방식": "", "비고": prep["error"]})
                st.markdown("---")
                continue

            with st.spinner(f"[{idx+1}/{len(jobs)}] {f.name} 분석 중..."):
                res = run_count(job["bytes"], job["offset"], prep_cfg, job["algo"], circularity_val, bg_opacity_val)

            count = res["count"]
            label = f"{job['algo']} (자동)" if is_auto else job["algo"]
            st.markdown(f"**카운트:** {count} 개  |  **방식:** `{label}`")
            if job["reason"]:
                st.caption(f"선택 근거: {job['reason']}")
            st.caption(
                f"{prep['px_per_mm']:.1f} px/mm · 콜로니 {'밝음' if prep['bright'] else '어두움'} · "
                f"대비 기준 {prep['thr']:.0f} · 네임펜 제외 {prep['excluded_ratio']:.1%}"
            )

            if count > TNTC_LIMIT:
                status = "TNTC"
                st.markdown(f"<span style='color:#ff4b4b; font-size:0.9em; font-weight:bold;'>🚨 TNTC (>{TNTC_LIMIT}, 참고용)</span>", unsafe_allow_html=True)
            elif count < TFTC_LIMIT:
                status = "TFTC"
                st.markdown(f"<span style='color:#d48806; font-size:0.9em;'>ℹ️ {TFTC_LIMIT}개 미만 (TFTC, 통계적 신뢰도 낮음)</span>", unsafe_allow_html=True)
            else:
                status = "계수 범위"
            if res["oversized"]:
                st.caption(f"🔴 최대 지름 초과로 제외된 덩어리: {res['oversized']}개")
            for w in prep["warnings"]:
                st.warning(w)

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(res["original_rgb"], caption="원본", width="stretch")
            with img_col2:
                st.image(res["result_rgb"], caption="결과 (🟡네임펜 제외, 🔴크기 초과)", width="stretch")
            st.markdown("---")

            rows.append({
                "파일명": f.name,
                "카운트": count,
                "방식": label,
                "판정": status,
                "선택 근거": job["reason"],
                "px/mm": round(prep["px_per_mm"], 2),
                "콜로니 밝기": "밝음" if prep["bright"] else "어두움",
                "대비 기준": round(prep["thr"], 1),
                "네임펜 제외 비율": round(prep["excluded_ratio"], 4),
                "크기 초과 제외": res["oversized"],
                "최소 지름(mm)": min_d_mm,
                "최대 지름(mm)": max_d_mm,
                "영역 크기": edge_crop_val,
                "비고": " / ".join(prep["warnings"]),
            })

    if rows:
        buf = io.StringIO()
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "📥 결과 CSV 다운로드",
            data=buf.getvalue().encode("utf-8-sig"),   # 엑셀 한글 깨짐 방지
            file_name="colony_counts.csv",
            mime="text/csv",
        )
else:
    st.info("👈 좌측 사이드바에서 분석 모드를 확인하고, 이미지를 업로드해 주세요.")
