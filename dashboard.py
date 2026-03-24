"""Interactive Streamlit dashboard for thermal monitoring demo."""

import base64
import io as _io
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

# Load .env
load_dotenv(Path(__file__).parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alerts.telegram_notifier import TelegramNotifier
from core.object_tracker import ThermalObjectTracker
from core.temperature_extractor import TemperatureExtractor
from core.thermal_detector import ThermalDetector
from core.video_processor import VideoProcessor
from utils.config import Config
from visualization.video_renderer import VideoRenderer

# Page configuration
st.set_page_config(
    page_title="Thermal Monitoring POC",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .warning-alert {
        background-color: #FFA500;
        color: white;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: left;
        font-size: 0.9rem;
        border-left: 5px solid #FF8C00;
    }
    .critical-alert {
        background-color: #FF0000;
        color: white;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: left;
        font-weight: bold;
        font-size: 0.9rem;
        border-left: 5px solid #8B0000;
    }
    .detection-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Session state defaults ---
_defaults = {
    "processing": False,
    "paused": False,
    "processed": False,
    "frame_idx": 0,
    "detections_log": [],
    "tracked_objects": {},
    "last_object_id": -1,
    "max_temp_overall": 0.0,
    "total_frames": 0,
    "last_frame_rgb": None,
    "current_max_temp": 0.0,
    "active_count": 0,
    "total_count": 0,
    "telegram_sent": False,
    "temp_history": [],
    "video_fps": 25.0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Header
st.markdown(
    '<p class="main-header">🔥 Sistema de Monitoreo Térmico - POC</p>',
    unsafe_allow_html=True,
)
st.markdown("### Detección de Polines Sobrecalentados en Fajas Transportadoras")

# --- Sidebar - Configuration ---
st.sidebar.header("⚙️ Configuración")

config_path = Path(__file__).parent / "config" / "config.yaml"
config = Config.load(str(config_path))

# Video source: existing files or upload
video_source = st.sidebar.radio(
    "Origen del video",
    ["📁 Videos existentes", "📤 Subir video"],
    disabled=st.session_state.processing,
)

if video_source == "📁 Videos existentes":
    video_files = list(Path(__file__).parent.glob("data/input/*.mp4"))
    if video_files:
        video_names = [v.name for v in video_files]
        selected_video = st.sidebar.selectbox(
            "Seleccionar video", video_names, index=0,
            disabled=st.session_state.processing,
        )
        video_path = Path(__file__).parent / "data" / "input" / selected_video
    else:
        st.sidebar.warning("No se encontraron videos en data/input/")
        video_path = None
else:
    uploaded_file = st.sidebar.file_uploader(
        "Arrastra o selecciona un video",
        type=["mp4", "avi", "mov"],
        disabled=st.session_state.processing,
    )
    if uploaded_file is not None:
        # Save uploaded file to a temp location
        if "uploaded_video_path" not in st.session_state or st.session_state.get("uploaded_video_name") != uploaded_file.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded_file.read())
            tmp.close()
            st.session_state.uploaded_video_path = tmp.name
            st.session_state.uploaded_video_name = uploaded_file.name
        video_path = st.session_state.uploaded_video_path
        selected_video = uploaded_file.name
        st.sidebar.success(f"✅ {uploaded_file.name}")
    else:
        video_path = None
        selected_video = None

if video_path is None:
    st.sidebar.info("Selecciona o sube un video para comenzar.")
    st.stop()

# Temperature range
st.sidebar.markdown("#### 🌡️ Rango de Temperatura")
temp_range_min = st.sidebar.number_input(
    "Temp. Mínima del Rango (°C)", min_value=0.0, max_value=100.0,
    value=float(config.thermal.get("temp_range_min", 20)), step=5.0,
    disabled=st.session_state.processing,
)
temp_range_max = st.sidebar.number_input(
    "Temp. Máxima del Rango (°C)", min_value=50.0, max_value=500.0,
    value=float(config.thermal.get("temp_range_max", 300)), step=10.0,
    disabled=st.session_state.processing,
)

st.sidebar.markdown("#### 🚨 Umbrales de Detección")
temp_threshold = st.sidebar.slider(
    "Umbral de Temperatura (°C)", min_value=20.0, max_value=300.0,
    value=float(config.detection["temperature_threshold"]), step=5.0,
    disabled=st.session_state.processing,
)
critical_threshold = st.sidebar.slider(
    "Umbral Crítico (°C)", min_value=temp_threshold + 5, max_value=350.0,
    value=max(float(config.detection["critical_threshold"]), temp_threshold + 5),
    step=5.0, disabled=st.session_state.processing,
)
min_area = st.sidebar.slider(
    "Área Mínima (píxeles)", min_value=5, max_value=500,
    value=int(config.detection["min_detection_area"]), step=5,
    disabled=st.session_state.processing,
)

# Telegram alerts toggle
st.sidebar.markdown("#### 📲 Alertas Telegram")
telegram_enabled = st.sidebar.toggle(
    "Enviar alertas por Telegram",
    value=True,
    disabled=st.session_state.processing,
)

# Belt inspection speed (Feature 4)
st.sidebar.markdown("#### 🚂 Simulacion de Posicion")
belt_speed_kmh = st.sidebar.number_input(
    "Velocidad del carro (km/h)", min_value=0.1, max_value=20.0,
    value=2.0, step=0.1,
    disabled=st.session_state.processing,
)

# Frames per batch — higher = smoother (fragment reruns are lightweight)
FRAMES_PER_BATCH = 60

# --- Control buttons ---
st.sidebar.markdown("---")

if not st.session_state.processing:
    if st.sidebar.button("🚀 Iniciar Monitoreo", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.session_state.paused = False
        st.session_state.processed = False
        st.session_state.frame_idx = 0
        st.session_state.detections_log = []
        st.session_state.tracked_objects = {}
        st.session_state.last_object_id = -1
        st.session_state.max_temp_overall = 0.0
        st.session_state.last_frame_rgb = None
        st.session_state.current_max_temp = 0.0
        st.session_state.active_count = 0
        st.session_state.total_count = 0
        st.session_state.telegram_sent = False
        st.session_state.temp_history = []
        # Store config snapshot
        st.session_state.cfg_threshold = temp_threshold
        st.session_state.cfg_critical = critical_threshold
        st.session_state.cfg_min_area = min_area
        st.session_state.cfg_temp_min = temp_range_min
        st.session_state.cfg_temp_max = temp_range_max
        st.session_state.cfg_video = str(video_path)
        st.session_state.cfg_telegram = telegram_enabled
        st.session_state.cfg_belt_speed = belt_speed_kmh
        st.rerun()
else:
    col_pause, col_stop = st.sidebar.columns(2)
    with col_pause:
        if st.session_state.paused:
            if st.button("▶️ Reanudar", use_container_width=True):
                st.session_state.paused = False
                st.rerun()
        else:
            if st.button("⏸️ Pausa", use_container_width=True):
                st.session_state.paused = True
                st.rerun()
    with col_stop:
        if st.button("⏹️ Detener", use_container_width=True):
            st.session_state.processing = False
            st.session_state.paused = False
            st.session_state.processed = True
            st.rerun()

# About
st.sidebar.markdown("---")
st.sidebar.info("""
**Sistema de Monitoreo Térmico POC**

Detecta automáticamente objetos sobrecalentados en videos termográficos.
""")


# --- Helper: generate thumbnail from BGR frame ---
def _frame_to_thumbnail_b64(frame_bgr, max_size=120):
    h, w = frame_bgr.shape[:2]
    scale = max_size / max(h, w)
    thumb = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf.tobytes()).decode()


# --- Helper: generate PDF report ---
def _generate_pdf_report(
    video_name, total_frames, total_objects, total_alerts, critical_count,
    max_temp, threshold, critical_threshold, tracked_objects, detections_log,
    temp_history, video_fps, belt_speed,
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Reporte de Monitoreo Termico", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Video: {video_name}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 8, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Resumen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Total Frames: {total_frames}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Objetos Unicos: {total_objects}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Total Alertas: {total_alerts}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Alertas Criticas: {critical_count}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Temperatura Maxima: {max_temp:.1f} C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Umbral Alerta: {threshold:.1f} C | Umbral Critico: {critical_threshold:.1f} C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Temperature chart
    if len(temp_history) >= 2:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Temperatura Maxima por Momento", new_x="LMARGIN", new_y="NEXT")

        step = max(1, len(temp_history) // 200)
        sampled = temp_history[::step]
        times = [s["time"] for s in sampled]
        temps = [s["temp"] for s in sampled]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(times, temps, color="#FF4B4B", linewidth=1.5)
        ax.fill_between(times, temps, alpha=0.15, color="#FF4B4B")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Temp C")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pdf.image(buf, x=15, w=180)
        pdf.ln(5)

    # Detected objects table
    if tracked_objects:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Objetos Detectados", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 9)
        col_widths = [15, 30, 30, 30, 25, 25, 35]
        headers = ["ID", "Temp Max", "Severidad", "Frames", "Duracion", "Detecciones", "Posicion"]
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 8, h, border=1, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        for oid, info in tracked_objects.items():
            dur = info["last_seen_frame"] - info["first_seen_frame"] + 1
            pos_first = (info["first_seen_frame"] / video_fps) * (belt_speed / 3600) if video_fps > 0 else 0
            pdf.cell(col_widths[0], 7, str(oid), border=1, align="C")
            pdf.cell(col_widths[1], 7, f"{info['max_temperature']:.1f} C", border=1, align="C")
            pdf.cell(col_widths[2], 7, info["severity"].upper(), border=1, align="C")
            pdf.cell(col_widths[3], 7, f"{info['first_seen_frame']}-{info['last_seen_frame']}", border=1, align="C")
            pdf.cell(col_widths[4], 7, f"{dur} frames", border=1, align="C")
            pdf.cell(col_widths[5], 7, str(info["total_detections"]), border=1, align="C")
            pdf.cell(col_widths[6], 7, f"Km {pos_first:.2f}", border=1, align="C")
            pdf.ln()
        pdf.ln(5)

    # Alert list with thumbnails
    if detections_log:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Listado de Alertas", new_x="LMARGIN", new_y="NEXT")

        for alert in detections_log:
            pdf.set_font("Helvetica", "B", 10)
            sev_label = "CRITICO" if alert["severity"] == "critical" else "ADVERTENCIA"
            pdf.cell(0, 7, f"Objeto #{alert['object_id']} - {sev_label}",
                     new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "", 9)
            pos = alert.get("position_km", 0.0)
            pdf.cell(0, 6, f"Frame: {alert['frame']} | Tiempo: {alert['timestamp']} | "
                           f"Temp: {alert['max_temp']:.1f} C | Posicion: Km {pos:.2f}",
                     new_x="LMARGIN", new_y="NEXT")

            if "frame_b64" in alert:
                try:
                    img_bytes = base64.b64decode(alert["frame_b64"])
                    img_buf = _io.BytesIO(img_bytes)
                    pdf.image(img_buf, w=40)
                except Exception:
                    pass

            pdf.ln(3)

    return pdf.output()


# =====================================================================
# Fragment: all dynamic content (video, metrics, chart, alerts)
# Only this section reruns between batches — sidebar & header stay stable.
# =====================================================================
@st.fragment
def monitoring_display():
    """Fragment that handles video display, metrics, chart, and alerts."""
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("### 📹 Video en Tiempo Real")
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

    with col2:
        st.markdown("### 📊 Métricas")

        # Traffic light indicator (Feature 6)
        traffic_light_placeholder = st.empty()

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            metric_temp = st.empty()
        with col2_2:
            metric_detections = st.empty()
        metric_progress = st.empty()

        # Temperature chart
        st.markdown("#### 🌡️ Temperatura máxima por momento")
        chart_placeholder = st.empty()

        st.markdown("---")
        st.markdown(
            '<p class="detection-header">🚨 Alertas en Tiempo Real</p>',
            unsafe_allow_html=True,
        )
        alerts_placeholder = st.empty()

    # --- Helpers ---
    def _render_traffic_light():
        t = st.session_state.current_max_temp
        cfg_crit = st.session_state.get("cfg_critical", 50.0)
        if t >= cfg_crit:
            color, label = "#dc3545", "CRITICO"
        elif t > 0:
            color, label = "#ffc107", "ADVERTENCIA"
        else:
            color, label = "#28a745", "NORMAL"
        html = f'''
        <div style="text-align:center;margin-bottom:0.8rem;">
            <div style="display:inline-block;width:60px;height:60px;border-radius:50%;
                        background-color:{color};border:3px solid #333;
                        box-shadow:0 0 20px {color};"></div>
            <p style="font-weight:bold;color:{color};margin-top:0.3rem;font-size:1.1rem;">{label}</p>
        </div>
        '''
        traffic_light_placeholder.markdown(html, unsafe_allow_html=True)

    def _render_alerts():
        if not st.session_state.detections_log:
            return
        recent = st.session_state.detections_log[-10:]
        html = ""
        for alert in reversed(recent):
            cls = "critical-alert" if alert["severity"] == "critical" else "warning-alert"
            icon = "🔴" if alert["severity"] == "critical" else "🟡"
            bbox = alert.get("bbox", (0, 0, 0, 0))
            pos = alert.get("position_km", 0.0)
            # Thumbnail (Feature 3)
            thumb_html = ""
            if "frame_b64" in alert:
                thumb_html = (
                    f'<img src="data:image/jpeg;base64,{alert["frame_b64"]}" '
                    f'style="width:90px;border-radius:4px;float:right;margin-left:8px;">'
                )
            html += f"""
            <div class="{cls}">
                {thumb_html}
                {icon} <strong>OBJETO DETECTADO</strong><br>
                ID: #{alert["object_id"]} | Frame {alert["frame"]} - {alert["timestamp"]}<br>
                Temperatura: {alert["max_temp"]:.1f}°C<br>
                Posicion: Km {pos:.2f} | Tamaño: {bbox[2]}x{bbox[3]}px
            </div>
            """
        alerts_placeholder.markdown(html, unsafe_allow_html=True)

    def _show_frame(frame_rgb):
        # Upscale small frames so they fill the column
        h, w = frame_rgb.shape[:2]
        if w < 480:
            scale = 480 // w + 1
            frame_rgb = cv2.resize(
                frame_rgb, (w * scale, h * scale),
                interpolation=cv2.INTER_NEAREST,
            )
        _, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        b64 = base64.b64encode(buf.tobytes()).decode()
        html = f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:8px;">'
        video_placeholder.markdown(html, unsafe_allow_html=True)
        return b64

    def _update_metrics():
        with metric_temp:
            t = st.session_state.current_max_temp
            cfg_crit = st.session_state.get("cfg_critical", 50.0)
            if t > 0:
                icon = "🔴" if t > cfg_crit else "🟡"
                st.metric("🌡️ Temp. Actual", f"{t:.1f}°C", delta=icon)
            else:
                st.metric("🌡️ Temp. Actual", "Normal", delta="✅")
        with metric_detections:
            st.metric(
                "🎯 Objetos Únicos",
                f"{st.session_state.total_count}",
                delta=f"Activos: {st.session_state.active_count}",
            )
        with metric_progress:
            total = st.session_state.total_frames or 1
            st.metric("⏱️ Frame", f"{st.session_state.frame_idx}/{total}")

    def _render_chart():
        if len(st.session_state.temp_history) < 2:
            return
        history = st.session_state.temp_history
        step = max(1, len(history) // 200)
        sampled = history[::step]
        times = [s["time"] for s in sampled]
        temps = [s["temp"] for s in sampled]

        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.plot(times, temps, color="#FF4B4B", linewidth=1.5)
        ax.fill_between(times, temps, alpha=0.15, color="#FF4B4B")
        ax.set_xlabel("Tiempo (s)", fontsize=8)
        ax.set_ylabel("Temp °C", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode()
        chart_placeholder.markdown(
            f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;">',
            unsafe_allow_html=True,
        )

    # --- Show last frame between reruns ---
    if st.session_state.last_frame_rgb is not None:
        video_placeholder.markdown(
            f'<img src="data:image/jpeg;base64,{st.session_state.last_frame_rgb}" style="width:100%;border-radius:8px;">',
            unsafe_allow_html=True,
        )
        if st.session_state.total_frames > 0:
            progress_bar.progress(
                min(st.session_state.frame_idx / st.session_state.total_frames, 1.0)
            )
        _render_traffic_light()
        _update_metrics()
        _render_chart()
        _render_alerts()

    # --- Paused state ---
    if st.session_state.processing and st.session_state.paused:
        status_text.warning(f"⏸️ En pausa — Frame {st.session_state.frame_idx}")

    # --- Process a batch of frames ---
    if st.session_state.processing and not st.session_state.paused:
        _thermal = {
            "temp_range_min": st.session_state.cfg_temp_min,
            "temp_range_max": st.session_state.cfg_temp_max,
        }
        _detection = {
            "temperature_threshold": st.session_state.cfg_threshold,
            "critical_threshold": st.session_state.cfg_critical,
            "min_detection_area": st.session_state.cfg_min_area,
        }

        temp_extractor = TemperatureExtractor(_thermal)
        detector = ThermalDetector(_detection)
        tracker = ThermalObjectTracker(
            max_disappeared=config.get("tracking.max_disappeared", 75),
            max_distance=config.get("tracking.max_distance", 150.0),
            min_confirm_frames=config.get("tracking.min_confirm_frames", 5),
        )
        renderer = VideoRenderer(config.visualization)
        video_proc = VideoProcessor(config.video)

        try:
            video_proc.load(st.session_state.cfg_video)
            video_info = video_proc.get_video_info()
            st.session_state.total_frames = video_info["total_frames"]
            fps = video_info["fps"]
            st.session_state.video_fps = fps

            # Seek to current frame position
            frame_idx = st.session_state.frame_idx
            if frame_idx > 0:
                video_proc.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            belt_speed = st.session_state.get("cfg_belt_speed", 2.0)
            frames_processed = 0
            finished = False

            while frames_processed < FRAMES_PER_BATCH:
                ret, frame = video_proc.read_frame()
                if not ret:
                    finished = True
                    break

                temp_frame = temp_extractor.extract_from_rgb(frame)
                detections, hot_mask = detector.detect(temp_frame)
                tracked = tracker.update(detections, frame_idx)

                # Check for new confirmed objects
                new_objects = {
                    obj_id: info
                    for obj_id, info in tracked.items()
                    if obj_id > st.session_state.last_object_id
                }

                # Update stats
                current_max_temp = 0.0
                if detections:
                    current_max_temp = detections[0]["max_temperature"]
                    st.session_state.max_temp_overall = max(
                        st.session_state.max_temp_overall, current_max_temp
                    )
                st.session_state.current_max_temp = current_max_temp

                # Record temperature for chart
                if detections:
                    chart_temp = detections[0]["max_temperature"]
                else:
                    mask = temp_frame[35:-15, 8:]
                    chart_temp = float(np.percentile(mask, 95))
                st.session_state.temp_history.append({
                    "frame": frame_idx,
                    "time": frame_idx / fps if fps > 0 else 0,
                    "temp": chart_temp,
                })

                # Compute timestamp from frame index (consistent across batches)
                if fps > 0:
                    _secs = frame_idx / fps
                    _m, _s = int(_secs // 60), int(_secs % 60)
                    _ms = int((_secs % 1) * 1000)
                    frame_timestamp = f"{_m:02d}:{_s:02d}.{_ms:03d}"
                else:
                    frame_timestamp = "00:00.000"

                # Compute belt position (Feature 4)
                position_km = (frame_idx / fps) * (belt_speed / 3600) if fps > 0 else 0.0

                # Render frame with overlays (no timestamp overlay)
                output_frame = renderer.render(
                    frame, temp_frame, detections, timestamp=""
                )

                if new_objects:
                    for obj_id, obj_info in new_objects.items():
                        bbox = obj_info["bbox"]
                        timestamp = frame_timestamp
                        # Capture thumbnail (Feature 3)
                        frame_b64_thumb = _frame_to_thumbnail_b64(output_frame)
                        st.session_state.detections_log.append({
                            "object_id": obj_id,
                            "frame": frame_idx,
                            "timestamp": timestamp,
                            "max_temp": obj_info["max_temperature"],
                            "severity": obj_info["severity"],
                            "bbox": bbox,
                            "type": "new_object",
                            "position_km": position_km,
                            "frame_b64": frame_b64_thumb,
                        })
                        st.session_state.last_object_id = obj_id

                        # Send Telegram alert (first detection only)
                        if st.session_state.get("cfg_telegram", False) and not st.session_state.telegram_sent:
                            _bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
                            _chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
                            if _bot_token and _chat_id:
                                _tg = TelegramNotifier(_bot_token, _chat_id)
                                sent = _tg.send_alert(
                                    frame_bgr=output_frame,
                                    object_id=obj_id,
                                    temperature=obj_info["max_temperature"],
                                    severity=obj_info["severity"],
                                    bbox=bbox,
                                    timestamp=timestamp,
                                    frame_number=frame_idx,
                                )
                                if sent:
                                    st.session_state.telegram_sent = True

                st.session_state.tracked_objects = tracker.get_all_objects()
                active, total = tracker.get_object_count()
                st.session_state.active_count = active
                st.session_state.total_count = total

                frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                b64 = _show_frame(frame_rgb)
                st.session_state.last_frame_rgb = b64
                progress_bar.progress(
                    (frame_idx + 1) / st.session_state.total_frames
                )

                _render_traffic_light()
                _update_metrics()
                # Render chart every 10 frames (matplotlib is expensive)
                if frames_processed % 10 == 0 or frames_processed == FRAMES_PER_BATCH - 1:
                    _render_chart()
                _render_alerts()

                status_text.info(
                    f"🔄 Procesando: Frame {frame_idx + 1}/{st.session_state.total_frames}"
                )

                frame_idx += 1
                frames_processed += 1

            video_proc.release()
            st.session_state.frame_idx = frame_idx

            if finished:
                st.session_state.processing = False
                st.session_state.processed = True
                status_text.success("✅ ¡Monitoreo completado!")
                progress_bar.progress(1.0)
                with metric_temp:
                    st.metric(
                        "🌡️ Temp. Máxima",
                        f"{st.session_state.max_temp_overall:.1f}°C",
                    )
                with metric_detections:
                    st.metric("🎯 Objetos Únicos", st.session_state.total_count)
                with metric_progress:
                    st.metric(
                        "📊 Total Alertas",
                        len(st.session_state.detections_log),
                    )
            else:
                # Continue processing — only this fragment reruns
                st.rerun()

        except Exception as e:
            status_text.error(f"❌ Error: {str(e)}")
            st.exception(e)
            st.session_state.processing = False


# --- Render the fragment ---
monitoring_display()

# --- Results summary (outside fragment — only renders on full page rerun) ---
if st.session_state.processed and not st.session_state.processing:
    st.markdown("---")
    st.markdown("### 📈 Resumen de Resultados")

    r1, r2, r3, r4 = st.columns(4)

    total_objects = len(st.session_state.tracked_objects)
    total_alerts = len(st.session_state.detections_log)
    max_temp = (
        max(d["max_temp"] for d in st.session_state.detections_log)
        if st.session_state.detections_log else 0
    )
    critical_count = sum(
        1 for d in st.session_state.detections_log if d["severity"] == "critical"
    )

    with r1:
        st.metric("📊 Total Frames", st.session_state.frame_idx)
    with r2:
        st.metric("🎯 Objetos Únicos", total_objects)
    with r3:
        st.metric("🔴 Alertas Críticas", critical_count)
    with r4:
        st.metric("🌡️ Temp. Máxima", f"{max_temp:.1f}°C")

    # Object details
    if st.session_state.tracked_objects:
        st.markdown("---")
        st.markdown("### 🔍 Detalles de Objetos Detectados")

        fps_val = st.session_state.get("video_fps", 25)
        speed = st.session_state.get("cfg_belt_speed", 2.0)

        for obj_id, obj_info in st.session_state.tracked_objects.items():
            with st.expander(
                f"📦 Objeto #{obj_id} - {obj_info['max_temperature']:.1f}°C "
                f"({obj_info['severity'].upper()})"
            ):
                # Show thumbnail if available (Feature 3)
                det_entry = next(
                    (d for d in st.session_state.detections_log if d["object_id"] == obj_id),
                    None,
                )
                if det_entry and "frame_b64" in det_entry:
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{det_entry["frame_b64"]}" '
                        f'style="width:150px;border-radius:4px;margin-bottom:8px;">',
                        unsafe_allow_html=True,
                    )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(
                        f"**Primera detección:** Frame {obj_info['first_seen_frame']}"
                    )
                    st.write(
                        f"**Última detección:** Frame {obj_info['last_seen_frame']}"
                    )
                with c2:
                    dur = (
                        obj_info["last_seen_frame"]
                        - obj_info["first_seen_frame"]
                        + 1
                    )
                    st.write(f"**Duración:** {dur} frames")
                    st.write(
                        f"**Detecciones:** {obj_info['total_detections']} veces"
                    )
                with c3:
                    st.write(
                        f"**Temp. máxima:** {obj_info['max_temperature']:.1f}°C"
                    )
                    st.write(
                        f"**Temp. media:** {obj_info['mean_temperature']:.1f}°C"
                    )
                    # Belt position (Feature 4)
                    pos_first = (obj_info["first_seen_frame"] / fps_val) * (speed / 3600) if fps_val > 0 else 0
                    pos_last = (obj_info["last_seen_frame"] / fps_val) * (speed / 3600) if fps_val > 0 else 0
                    st.write(f"**Posición:** Km {pos_first:.2f} - Km {pos_last:.2f}")

    # Export
    if st.session_state.detections_log or st.session_state.tracked_objects:
        st.markdown("---")
        st.markdown("### 💾 Exportar Resultados")

        exp1, exp2, exp3 = st.columns(3)

        with exp1:
            report_data = {
                "metadata": {
                    "date": datetime.now().isoformat(),
                    "video": selected_video,
                    "threshold": st.session_state.get(
                        "cfg_threshold", temp_threshold
                    ),
                    "critical_threshold": st.session_state.get(
                        "cfg_critical", critical_threshold
                    ),
                    "total_frames": st.session_state.frame_idx,
                    "total_objects": total_objects,
                    "total_alerts": total_alerts,
                },
                "tracked_objects": [
                    {"object_id": oid, **info}
                    for oid, info in st.session_state.tracked_objects.items()
                ],
                "alerts": [
                    {k: v for k, v in d.items() if k != "frame_b64"}
                    for d in st.session_state.detections_log
                ],
            }
            st.download_button(
                label="📥 Descargar JSON",
                data=json.dumps(report_data, indent=2),
                file_name=f"thermal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with exp2:
            txt = f"""REPORTE DE MONITOREO TERMICO
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Video: {selected_video}

=== RESUMEN ===
Total Frames: {st.session_state.frame_idx}
Objetos Unicos: {total_objects}
Alertas: {total_alerts}
Alertas Criticas: {critical_count}
Temp. Maxima: {max_temp:.1f} C

=== OBJETOS DETECTADOS ===
"""
            for oid, info in st.session_state.tracked_objects.items():
                dur = info["last_seen_frame"] - info["first_seen_frame"] + 1
                txt += f"\nObjeto #{oid}:"
                txt += f"\n  Temp. Maxima: {info['max_temperature']:.1f} C"
                txt += f"\n  Severidad: {info['severity'].upper()}"
                txt += f"\n  Frames: {info['first_seen_frame']}-{info['last_seen_frame']} ({dur} frames)"
                txt += f"\n  Detecciones: {info['total_detections']}\n"

            st.download_button(
                label="📄 Descargar TXT",
                data=txt,
                file_name=f"thermal_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with exp3:
            pdf_bytes = _generate_pdf_report(
                video_name=selected_video,
                total_frames=st.session_state.frame_idx,
                total_objects=total_objects,
                total_alerts=total_alerts,
                critical_count=critical_count,
                max_temp=max_temp,
                threshold=st.session_state.get("cfg_threshold", temp_threshold),
                critical_threshold=st.session_state.get("cfg_critical", critical_threshold),
                tracked_objects=st.session_state.tracked_objects,
                detections_log=st.session_state.detections_log,
                temp_history=st.session_state.temp_history,
                video_fps=st.session_state.get("video_fps", 25),
                belt_speed=st.session_state.get("cfg_belt_speed", 2.0),
            )
            st.download_button(
                label="📕 Descargar PDF",
                data=pdf_bytes,
                file_name=f"thermal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p>🔥 Sistema de Monitoreo Térmico POC v1.0.0</p>
</div>
""",
    unsafe_allow_html=True,
)
