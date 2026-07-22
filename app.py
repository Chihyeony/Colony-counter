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
    /* 타이틀 폰트 및 여백 조정 */
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
    
    /* 버튼 호버 애니메이션 */
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


# --- 웹 UI 구성 ---
st.markdown('<div class="main-title">🧫 Auto Colony Counter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Watershed Algorithm 기반 자동 배지 콜로니 계수 및 분석 도구</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 분석 파라미터 설정")
    
    with st.expander("배지 테두리 마스킹", expanded=True):
        edge_crop_val = st.slider(
            "테두리 절삭 비율", 
            min_value=0.70, max_value=0.95, value=0.82, step=0.01,
            help="0.85-0.90을 권장합니다."
        )
        
        st.markdown("<div style='text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 10px;'>테두리 중심 미세 조정 (10px)</div>", unsafe_allow_html=True)
        move_step = 10 
        
        # '초기화' 텍스트가 두 줄로 꺾이지 않도록 가운데 열을 더 넓게(1.5 비율) 설정
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col1:
            if st.button("↖", use_container_width=True):
                st.session_state.x_offset -= move_step
                st.session_state.y_offset -= move_step
            if st.button("⬅", use_container_width=True):
                st.session_state.x_offset -= move_step
            if st.button("↙", use_container_width=True):
                st.session_state.x_offset -= move_step
                st.session_state.y_offset += move_step
                
        with col2:
            if st.button("⬆", use_container_width=True):
                st.session_state.y_offset -= move_step
            if st.button("초기화", use_container_width=True):
                st.session_state.x_offset = 0
                st.session_state.y_offset = 0
            if st.button("⬇", use_container_width=True):
                st.session_state.y_offset += move_step
                
        with col3:
            if st.button("↗", use_container_width=True):
                st.session_state.x_offset += move_step
                st.session_state.y_offset -= move_step
            if st.button("➡", use_container_width=True):
                st.session_state.x_offset += move_step
            if st.button("↘", use_container_width=True):
                st.session_state.x_offset += move_step
                st.session_state.y_offset += move_step
                
        st.caption(f"📍 현재 보정값: X({st.session_state.x_offset}), Y({st.session_state.y_offset})")
    
    with st.expander("노이즈 및 화면 설정", expanded=True):
        min_size_val = st.slider(
            "최소 콜로니 크기 (픽셀)", 
            min_value=0, max_value=200, value=15, step=1,
            help="80-120을 권장합니다."
        )
        bg_opacity_val = st.slider(
            "배경 투명도 (비침 정도)", 
            min_value=0.0, max_value=1.0, value=0.15, step=0.05
        )

uploaded_file = st.file_uploader("콜로니 이미지를 드래그 앤 드롭 하세요 (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner("알고리즘 분석 중..."):
        count, original_img, result_img = count_colonies_watershed(
            file_bytes, 
            edge_crop=edge_crop_val,
            x_offset=st.session_state.x_offset,
            y_offset=st.session_state.y_offset,
            min_size=min_size_val,
            bg_opacity=bg_opacity_val
        )
        
        st.markdown("---")
        
        if count > 1000:
            st.error(f"🚨 TNTC (Too Numerous To Count) 상태입니다.")
        else:
            st.metric(label="Total Colony Count", value=f"{count} 개")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 원본 이미지")
            st.image(original_img, use_container_width=True) 
            
        with col2:
            st.markdown("#### 분석 결과 (Watershed Mask)")
            st.image(result_img, use_container_width=True) 
else:
    st.info("👈 좌측 사이드바에서 분석 파라미터를 설정하고, 이미지를 업로드해 주세요.")