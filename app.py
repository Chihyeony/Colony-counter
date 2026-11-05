import streamlit as st
import cv2
import numpy as np
import math

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

if 'x_offset' not in st.session_state:
    st.session_state.x_offset = 0
if 'y_offset' not in st.session_state:
    st.session_state.y_offset = 0


# ---------------------------------------------------------
# 🧹 핵심 전처리 모듈 (빛 반사 제거 기능 제외, 네임펜만 타겟팅)
# ---------------------------------------------------------
def preprocess_image(file_bytes, edge_crop, x_offset, y_offset, remove_marker):
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    original_bgr = cv2.imdecode(img_array, 1)
    
    # 처리 속도 향상 및 Inpaint 메모리 최적화를 위한 리사이징 (최대 1500px)
    h, w = original_bgr.shape[:2]
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        original_bgr = cv2.resize(original_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        x_offset = int(x_offset * scale)
        y_offset = int(y_offset * scale)
        
    h, w = original_bgr.shape[:2]
    if w > h:
        original_bgr = cv2.rotate(original_bgr, cv2.ROTATE_90_CLOCKWISE)
        
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. 배지(Petri dish) 테두리 인식 및 마스크 생성
    blur_for_mask = cv2.GaussianBlur(gray, (51, 51), 0)
    _, mask_thresh = cv2.threshold(blur_for_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dish_mask = np.ones_like(gray) * 255 
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        dish_mask = np.zeros_like(gray)
        if len(largest_contour) >= 5:
            (x, y), (axes1, axes2), angle = cv2.fitEllipse(largest_contour)
            center = (x + x_offset, y + y_offset)
            new_axes = (axes1 * edge_crop, axes2 * edge_crop)
            cv2.ellipse(dish_mask, (center, new_axes, angle), 255, -1)
        else:
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            center = (int(x + x_offset), int(y + y_offset))
            cv2.circle(dish_mask, center, int(radius * edge_crop), 255, -1)
            
    # 2. 노이즈(네임펜 글씨 및 어두운 그림자) 추출 (HSV 색공간 활용)
    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    exclusion_mask = np.zeros_like(gray)
    
    if remove_marker:
        # 네임펜: 밝기(V)가 칠흑같이 어둡거나(검은 펜), 채도(S)가 뚜렷함(빨강, 파랑 등 유색 펜)
        marker_mask = cv2.bitwise_or(cv2.inRange(v_ch, 0, 50), cv2.inRange(s_ch, 110, 255))
        marker_mask = cv2.dilate(marker_mask, np.ones((5,5), np.uint8), iterations=2)
        exclusion_mask = cv2.bitwise_or(exclusion_mask, marker_mask)
        
    # 배지 바깥의 노이즈는 제외
    exclusion_mask = cv2.bitwise_and(exclusion_mask, dish_mask)
    
    # 3. OpenCV Inpaint 기법: 네임펜 영역을 주변 배경색으로 감쪽같이 덮어씌움
    if np.any(exclusion_mask):
        gray_clean = cv2.inpaint(gray, exclusion_mask, 3, cv2.INPAINT_TELEA)
    else:
        gray_clean = gray.copy()
        
    return original_bgr, gray_clean, dish_mask, exclusion_mask


# ---------------------------------------------------------
# 🤖 AI 기반 이미지 특성 분석 (스마트 알고리즘 추천)
# ---------------------------------------------------------
def analyze_image_for_algorithm(gray_clean, dish_mask):
    gray_masked = cv2.bitwise_and(gray_clean, gray_clean, mask=dish_mask)
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)
    
    mean_val = cv2.mean(blurred, mask=dish_mask)[0]
    temp_for_otsu = blurred.copy()
    temp_for_otsu[dish_mask == 0] = int(mean_val)
    _, thresh = cv2.threshold(temp_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, thresh, mask=dish_mask)
    
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_ctrs = [c for c in ctrs if cv2.contourArea(c) > 30]
    
    if not valid_ctrs: return "Blob" 
        
    circularities = []
    solidities = []
    
    for c in valid_ctrs:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        
        if perimeter > 0 and hull_area > 0:
            circularities.append(4 * np.pi * (area / (perimeter * perimeter)))
            solidities.append(area / hull_area)
            
    avg_circ = np.mean(circularities) if circularities else 0
    avg_solidity = np.mean(solidities) if solidities else 0
    contour_count = len(valid_ctrs)
    
    if avg_solidity < 0.85: return "Blob"
    if avg_circ < 0.75 and avg_solidity >= 0.85: return "Watershed"
    if contour_count > 100: return "Watershed"
    return "Blob"


# ---------------------------------------------------------
# 💧 1. Watershed 알고리즘
# ---------------------------------------------------------
def count_colonies_watershed(original_bgr, gray_clean, dish_mask, exclusion_mask, min_diameter, bg_opacity):
    gray_for_analysis = cv2.bitwise_and(gray_clean, gray_clean, mask=dish_mask)
    blurred = cv2.GaussianBlur(gray_for_analysis, (5, 5), 0)
    
    mean_val = cv2.mean(blurred, mask=dish_mask)[0]
    temp_for_otsu = blurred.copy()
    temp_for_otsu[dish_mask == 0] = int(mean_val)
    
    _, thresh = cv2.threshold(temp_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, thresh, mask=dish_mask)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    local_max = cv2.dilate(dist_transform, np.ones((5,5), np.uint8))
    sure_fg = np.uint8((dist_transform == local_max) & (dist_transform > 1.0)) * 255

    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original_bgr, markers)
    
    labels, counts = np.unique(markers, return_counts=True)
    min_area_calc = math.pi * ((min_diameter / 2) ** 2)
    valid_labels = labels[(counts >= min_area_calc) & (labels >= 2)]
    colony_count = len(valid_labels)
    
    valid_mask = np.isin(markers, valid_labels).astype(np.uint8) * 255
    c, _ = cv2.findContours(valid_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    final_boundary_mask = np.zeros_like(gray_clean)
    cv2.drawContours(final_boundary_mask, c, -1, 255, 1)
    thick_boundary = cv2.dilate(final_boundary_mask, np.ones((3,3), np.uint8), iterations=1)
    
    dark_bg = cv2.addWeighted(original_bgr, bg_opacity, np.zeros_like(original_bgr), 0, 0)
    mask_3ch = cv2.cvtColor(dish_mask, cv2.COLOR_GRAY2BGR)
    final_img = np.where(mask_3ch == 255, original_bgr, dark_bg)
    
    # 🟡 제외된 네임펜 영역 시각화 (노란색 반투명)
    overlay = final_img.copy()
    overlay[exclusion_mask == 255] = [0, 215, 255] 
    final_img = cv2.addWeighted(overlay, 0.4, final_img, 0.6, 0)
    
    final_img[thick_boundary > 0] = [0, 255, 0] 
    
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    return colony_count, original_rgb, final_rgb


# ---------------------------------------------------------
# 🔍 2. Blob Detection 알고리즘
# ---------------------------------------------------------
def count_colonies_blob(original_bgr, gray_clean, dish_mask, exclusion_mask, min_diameter, min_circularity, bg_opacity):
    gray_masked = cv2.bitwise_and(gray_clean, gray_clean, mask=dish_mask)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 40
    params.maxThreshold = 255
    params.thresholdStep = 10
    
    min_area_calc = math.pi * ((min_diameter / 2) ** 2)
    params.filterByArea = True
    params.minArea = max(1, min_area_calc) 
    params.maxArea = 500000 
    
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.5

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_masked)
    
    colony_count = len(keypoints)

    dark_bg = cv2.addWeighted(original_bgr, bg_opacity, np.zeros_like(original_bgr), 0, 0)
    mask_3ch = cv2.cvtColor(dish_mask, cv2.COLOR_GRAY2BGR)
    final_img = np.where(mask_3ch == 255, original_bgr, dark_bg)

    # 🟡 제외된 네임펜 영역 시각화 (노란색 반투명)
    overlay = final_img.copy()
    overlay[exclusion_mask == 255] = [0, 215, 255] 
    final_img = cv2.addWeighted(overlay, 0.4, final_img, 0.6, 0)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        cv2.circle(final_img, (x, y), r + 2, (0, 255, 0), 2)
        cv2.circle(final_img, (x, y), 1, (0, 0, 255), -1)
    
    boundary_contours, _ = cv2.findContours(dish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final_img, boundary_contours, -1, (255, 0, 0), 2)

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    return colony_count, original_rgb, final_rgb


# ---------------------------------------------------------
# 3. 웹 UI 구성
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧫 Auto Colony Counter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">스마트 분석이 탑재된 콜로니 카운터 MADE BY 치현</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 분석 방식 및 파라미터")
    
    selected_method = st.radio(
        "🧪 분석 알고리즘 선택",
        options=[
            "🌟 스마트 자동 선택 (이미지 분석 후 최적 적용)",
            "Watershed (콜로니 수가 많고 다닥다닥 겹쳤을 때)", 
            "Blob (콜로니 수가 적거나 노이즈가 많을 때)"
        ],
        index=0
    )
    st.markdown("---")
    
    # 🚀 네임펜 노이즈 제거 메뉴만 깔끔하게 유지
    with st.expander("🧹 특수 노이즈 제거", expanded=True):
        st.markdown("<p style='font-size:0.85em; color:gray;'>네임펜 글씨 등으로 인식된 영역은 🟡노란색으로 처리됩니다.</p>", unsafe_allow_html=True)
        remove_marker_val = st.checkbox("🖊️ 네임펜 무시 (글씨, 어두운 그림자)", value=True, help="배지 뒷면의 네임펜 글씨나 그림자를 지워버립니다.")
    
    with st.expander("✂️ 인식 영역 설정", expanded=False):
        edge_crop_val = st.slider("영역 크기 설정", 0.70, 0.95, 0.85, 0.01)
        bg_opacity_val = st.slider("배경 투명도", 0.0, 1.0, 0.15, 0.05)
        st.markdown("<div style='text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 10px;'>영역 위치 미세 조정 (10px 단위)</div>", unsafe_allow_html=True)
        move_step = 10 
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col1:
            if st.button("↖", width="stretch"): st.session_state.x_offset -= move_step; st.session_state.y_offset -= move_step
            if st.button("⬅", width="stretch"): st.session_state.x_offset -= move_step
            if st.button("↙", width="stretch"): st.session_state.x_offset -= move_step; st.session_state.y_offset += move_step
        with col2:
            if st.button("⬆", width="stretch"): st.session_state.y_offset -= move_step
            if st.button("초기화", width="stretch"): st.session_state.x_offset = 0; st.session_state.y_offset = 0
            if st.button("⬇", width="stretch"): st.session_state.y_offset += move_step
        with col3:
            if st.button("↗", width="stretch"): st.session_state.x_offset += move_step; st.session_state.y_offset -= move_step
            if st.button("➡", width="stretch"): st.session_state.x_offset += move_step
            if st.button("↘", width="stretch"): st.session_state.x_offset += move_step; st.session_state.y_offset += move_step
    
    watershed_min_diameter = 15 
    blob_min_diameter = 10      
    circularity_val = 0.6       

    if "Watershed" in selected_method or "스마트" in selected_method:
        with st.expander("💧 Watershed 세부 설정", expanded=("Watershed" in selected_method)):
            watershed_min_diameter = st.slider("Watershed 최소 지름 (px)", 1, 200, 15, 1)
            
    if "Blob" in selected_method or "스마트" in selected_method:
        with st.expander("🔍 Blob 세부 설정", expanded=("Blob" in selected_method)):
            blob_min_diameter = st.slider("Blob 최소 지름 (px)", 1, 200, 10, 1)
            circularity_val = st.slider("Blob 최소 원형도", 0.1, 1.0, 0.35, 0.05)


uploaded_files = st.file_uploader("콜로니 이미지를 드래그 앤 드롭 하세요 (.jpg, .png) - 최대 10장", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("⚠️ 최대 10개의 이미지만 분석합니다. 처음 10개만 처리됩니다.")
        uploaded_files = uploaded_files[:10]
        
    st.markdown(f"### 📊 총 {len(uploaded_files)}개의 이미지 분석 결과")
    st.markdown("---")
    
    grid_cols = st.columns(3)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()
        
        with st.spinner(f"[{idx+1}/{len(uploaded_files)}] {uploaded_file.name} 분석 중..."):
            
            # 빛 반사 제거 옵션은 제외하고 네임펜만 처리
            original_bgr, gray_clean, dish_mask, exclusion_mask = preprocess_image(
                file_bytes, edge_crop_val, st.session_state.x_offset, st.session_state.y_offset,
                remove_marker_val
            )
            
            if "스마트" in selected_method:
                target_algo = analyze_image_for_algorithm(gray_clean, dish_mask)
            elif "Watershed" in selected_method:
                target_algo = "Watershed"
            else:
                target_algo = "Blob"
            
            if target_algo == "Watershed":
                count, original_img, result_img = count_colonies_watershed(
                    original_bgr, gray_clean, dish_mask, exclusion_mask,
                    watershed_min_diameter, bg_opacity_val 
                )
                display_label = "🤖 Watershed (AI)" if "스마트" in selected_method else "Watershed"
            else:
                count, original_img, result_img = count_colonies_blob(
                    original_bgr, gray_clean, dish_mask, exclusion_mask,
                    blob_min_diameter, circularity_val, bg_opacity_val 
                )
                display_label = "🤖 Blob (AI)" if "스마트" in selected_method else "Blob"
            
            with grid_cols[idx % 3]:
                with st.container():
                    st.markdown(f"##### 🧪 {uploaded_file.name}")
                    st.markdown(f"**카운트:** {count} 개  |  **방식:** `{display_label}`")
                    
                    if count > 1000:
                        st.markdown("<span style='color:#ff4b4b; font-size:0.9em; font-weight:bold;'>🚨 TNTC (참고용)</span>", unsafe_allow_html=True)
                    
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.image(original_img, caption="원본", use_container_width=True) 
                    with img_col2:
                        st.image(result_img, caption="결과 (노란색=네임펜제거)", use_container_width=True) 
                        
                    st.markdown("---") 
else:
    st.info("👈 좌측 사이드바에서 분석 모드를 확인하고, 이미지를 업로드해 주세요.")