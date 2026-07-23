import streamlit as st
import cv2
import numpy as np

# UI 기본 설정 (넓은 화면, 탭 제목 및 아이콘 설정)
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

# 방향키 조작을 위해 X, Y 위치값을 기억하는 세션 상태 초기화
if 'x_offset' not in st.session_state:
    st.session_state.x_offset = 0
if 'y_offset' not in st.session_state:
    st.session_state.y_offset = 0


# ---------------------------------------------------------
# 1. Watershed 알고리즘
# ---------------------------------------------------------
def count_colonies_watershed(file_bytes, edge_crop=0.82, x_offset=0, y_offset=0, min_size=15, bg_opacity=0.2):
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    original_bgr = cv2.imdecode(img_array, 1)
    
    h, w = original_bgr.shape[:2]
    if w > h:
        original_bgr = cv2.rotate(original_bgr, cv2.ROTATE_90_CLOCKWISE)
        
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    blur_for_mask = cv2.GaussianBlur(gray, (51, 51), 0)
    _, mask_thresh = cv2.threshold(blur_for_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones_like(gray) * 255 
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        
        if len(largest_contour) >= 5:
            (x, y), (axes1, axes2), angle = cv2.fitEllipse(largest_contour)
            center = (x + x_offset, y + y_offset)
            new_axes = (axes1 * edge_crop, axes2 * edge_crop)
            cv2.ellipse(mask, (center, new_axes, angle), 255, -1)
        else:
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            center = (int(x + x_offset), int(y + y_offset))
            cv2.circle(mask, center, int(radius * edge_crop), 255, -1)

    gray_for_analysis = cv2.bitwise_and(gray, gray, mask=mask)
    blurred = cv2.GaussianBlur(gray_for_analysis, (5, 5), 0)
    mean_val = cv2.mean(blurred, mask=mask)[0]
    temp_for_otsu = blurred.copy()
    temp_for_otsu[mask == 0] = int(mean_val)
    
    _, thresh = cv2.threshold(temp_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

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
    
    valid_labels = labels[(counts >= min_size) & (labels >= 2)]
    colony_count = len(valid_labels)
    
    valid_mask = np.isin(markers, valid_labels).astype(np.uint8) * 255
    c, _ = cv2.findContours(valid_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    final_boundary_mask = np.zeros_like(gray)
    cv2.drawContours(final_boundary_mask, c, -1, 255, 1)
    thick_boundary = cv2.dilate(final_boundary_mask, np.ones((3,3), np.uint8), iterations=1)
    
    dark_bg = cv2.addWeighted(original_bgr, bg_opacity, np.zeros_like(original_bgr), 0, 0)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    final_img = np.where(mask_3ch == 255, original_bgr, dark_bg)
    final_img[thick_boundary > 0] = [0, 255, 0] 
    
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    return colony_count, original_rgb, final_rgb


# ---------------------------------------------------------
# 2. Blob Detection 알고리즘
# ---------------------------------------------------------
def count_colonies_blob(file_bytes, edge_crop, x_offset, y_offset, min_area, min_circularity, bg_opacity):
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    original_bgr = cv2.imdecode(img_array, 1)
    
    h, w = original_bgr.shape[:2]
    if w > h:
        original_bgr = cv2.rotate(original_bgr, cv2.ROTATE_90_CLOCKWISE)
        
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    blur_for_mask = cv2.GaussianBlur(gray, (51, 51), 0)
    _, mask_thresh = cv2.threshold(blur_for_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones_like(gray) * 255 
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        if len(largest_contour) >= 5:
            (x, y), (axes1, axes2), angle = cv2.fitEllipse(largest_contour)
            center = (x + x_offset, y + y_offset)
            new_axes = (axes1 * edge_crop, axes2 * edge_crop)
            cv2.ellipse(mask, (center, new_axes, angle), 255, -1)
        else:
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            center = (int(x + x_offset), int(y + y_offset))
            cv2.circle(mask, center, int(radius * edge_crop), 255, -1)

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 40
    params.maxThreshold = 255
    params.thresholdStep = 10
    
    # 🚀 크기 설정 (최대 크기는 10000으로 내부 고정)
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = 10000 
    
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
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    final_img = np.where(mask_3ch == 255, original_bgr, dark_bg)

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        cv2.circle(final_img, (x, y), r + 2, (0, 255, 0), 2)
        cv2.circle(final_img, (x, y), 1, (0, 0, 255), -1)
    
    boundary_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(final_img, boundary_contours, -1, (255, 0, 0), 2)

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    return colony_count, original_rgb, final_rgb


# ---------------------------------------------------------
# 3. 웹 UI 구성
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧫 Auto Colony Counter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Watershed / Blob Detection 알고리즘을 이용한 콜로니 카운터 MADE BY 치현</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 분석 방식 및 파라미터")
    
    # 🚀 이름이 직관적으로 변경된 알고리즘 선택 
    selected_method = st.radio(
        "🧪 분석 알고리즘 선택",
        options=[
            "Watershed (콜로니 수가 많고 다닥다닥 겹친 콜로니가 많을 때)", 
            "Blob (콜로니 수가 적거나 스크래치, 빛 번짐등 노이즈가 많을 때)"
        ],
        index=0,
        help="업로드한 배지의 상태(밀집도, 노이즈 유무)에 따라 더 결과가 좋은 방식을 선택하세요."
    )
    st.markdown("---")
    
    with st.expander("✂️ 인식 영역 설정", expanded=True):
        edge_crop_val = st.slider(
            "영역 크기 설정", 0.70, 0.95, 0.85, 0.01,
            help="영역 내의 콜로니만 카운팅합니다. 배지 테두리의 빛 반사가 콜로니로 인식될 때 수치를 낮추세요."
        )
        bg_opacity_val = st.slider(
                    "배경 투명도", 0.0, 1.0, 0.15, 0.05,
                    help="0.0이면 완전 검은색, 1.0이면 원본 밝기 그대로 표시됩니다."
                )
        st.markdown("<div style='text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 10px;'>영역 위치 미세 조정 (10px 단위)</div>", unsafe_allow_html=True)
        move_step = 10 
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col1:
            if st.button("↖", use_container_width=True, help="왼쪽 위로 이동"): st.session_state.x_offset -= move_step; st.session_state.y_offset -= move_step
            if st.button("⬅", use_container_width=True, help="왼쪽으로 이동"): st.session_state.x_offset -= move_step
            if st.button("↙", use_container_width=True, help="왼쪽 아래로 이동"): st.session_state.x_offset -= move_step; st.session_state.y_offset += move_step
        with col2:
            if st.button("⬆", use_container_width=True, help="위로 이동"): st.session_state.y_offset -= move_step
            if st.button("초기화", use_container_width=True, help="중심점을 원상복구합니다."): st.session_state.x_offset = 0; st.session_state.y_offset = 0
            if st.button("⬇", use_container_width=True, help="아래로 이동"): st.session_state.y_offset += move_step
        with col3:
            if st.button("↗", use_container_width=True, help="오른쪽 위로 이동"): st.session_state.x_offset += move_step; st.session_state.y_offset -= move_step
            if st.button("➡", use_container_width=True, help="오른쪽으로 이동"): st.session_state.x_offset += move_step
            if st.button("↘", use_container_width=True, help="오른쪽 아래로 이동"): st.session_state.x_offset += move_step; st.session_state.y_offset += move_step
    
    # 🚀 공통 파라미터 탭
    with st.expander("🎛️ 공통 설정 (크기 및 화면)", expanded=True):
        common_min_size = st.slider(
            "콜로니 인식 최소 크기", 1, 100, 10, 1,
            help="이 수치보다 작은 개체는 단순 먼지나 찌꺼기로 간주하여 카운트하지 않습니다."
        )
        

    # 🚀 Blob 선택 시에만 나타나는 전용 파라미터 탭
    circularity_val = 0.6 # Watershed 선택 시에도 기본 변수 할당 에러 방지
    if "Blob" in selected_method:
        with st.expander("🔍 Blob 방식 전용 설정", expanded=True):
            circularity_val = st.slider(
                "콜로니 인식 최소 원형도", 0.1, 1.0, 0.35, 0.05,
                help="0.1에 가까울수록 찌그러진 모양도 콜로니로 인식하고, 1.0에 가까울수록 완벽한 동그라미 모양만 콜로니로 간주합니다."
            )

uploaded_file = st.file_uploader("콜로니 이미지를 드래그 앤 드롭 하세요 (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner("선택된 알고리즘으로 분석 중..."):
        if "Watershed" in selected_method:
            count, original_img, result_img = count_colonies_watershed(
                file_bytes, 
                edge_crop=edge_crop_val,
                x_offset=st.session_state.x_offset,
                y_offset=st.session_state.y_offset,
                min_size=common_min_size, # 공통 변수 연결
                bg_opacity=bg_opacity_val # 공통 변수 연결
            )
            marker_text = "Watershed Mask"
        else:
            count, original_img, result_img = count_colonies_blob(
                file_bytes, 
                edge_crop=edge_crop_val,
                x_offset=st.session_state.x_offset,
                y_offset=st.session_state.y_offset,
                min_area=common_min_size, # 공통 변수 연결
                min_circularity=circularity_val,
                bg_opacity=bg_opacity_val # 공통 변수 연결
            )
            marker_text = "Blob 마커"
        
        st.markdown("---")
        
        st.metric(label=f"Total Colony Count ({'Watershed' if 'Watershed' in selected_method else 'Blob'})", value=f"{count} 개")
        
        if count > 1000:
            st.error("🚨 TNTC (Too Numerous To Count) 입니다. 표시된 개수는 참고용으로만 활용하세요.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 원본 이미지")
            st.image(original_img, use_container_width=True) 
            
        with col2:
            st.markdown(f"#### 분석 결과 ({marker_text})")
            st.image(result_img, use_container_width=True) 
else:
    st.info("👈 좌측 사이드바에서 분석 모드를 확인하고, 이미지를 업로드해 주세요.")