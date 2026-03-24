"""Interactive Streamlit dashboard for thermal monitoring demo."""

import base64
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv

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
    "processed": False,
    "processing": False,
    "detections_log": [],
    "tracked_objects": {},
    "max_temp_overall": 0.0,
    "total_frames": 0,
    "temp_history": [],
    "output_video_path": None,
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

# Video selection
video_files = list(Path(__file__).parent.glob("data/input/*.mp4"))
if video_files:
    video_names = [v.name for v in video_files]
    selected_video = st.sidebar.selectbox(
        "Seleccionar video", video_names, index=0,
        disabled=st.session_state.processing,
    )
    video_path = Path(__file__).parent / "data" / "input" / selected_video
else:
    st.sidebar.error("No se encontraron videos en data/input/")
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

# --- Control buttons ---
st.sidebar.markdown("---")

if not st.session_state.processing and not st.session_state.processed:
    if st.sidebar.button("🚀 Iniciar Monitoreo", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.session_state.processed = False
        st.session_state.detections_log = []
        st.session_state.tracked_objects = {}
        st.session_state.max_temp_overall = 0.0
        st.session_state.temp_history = []
        st.session_state.output_video_path = None
        st.rerun()

if st.session_state.processed:
    if st.sidebar.button("🔄 Nuevo Análisis", use_container_width=True):
        st.session_state.processed = False
        st.session_state.processing = False
        st.session_state.output_video_path = None
        st.rerun()

# About
st.sidebar.markdown("---")
st.sidebar.info("""
**Sistema de Monitoreo Térmico POC**

Detecta automáticamente objetos sobrecalentados en videos termográficos.
""")


# =====================================================================
# PHASE 1: Pre-process the entire video
# =====================================================================
if st.session_state.processing and not st.session_state.processed:
    st.markdown("### ⏳ Procesando video...")
    st.markdown("Analizando frames y detectando anomalías térmicas. Esto puede tardar unos segundos.")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Build components
    _thermal = {"temp_range_min": temp_range_min, "temp_range_max": temp_range_max}
    _detection = {
        "temperature_threshold": temp_threshold,
        "critical_threshold": critical_threshold,
        "min_detection_area": min_area,
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
        video_proc.load(str(video_path))
        video_info = video_proc.get_video_info()
        total_frames = video_info["total_frames"]
        fps = video_info["fps"]

        # Create temp output video
        tmp_dir = tempfile.mkdtemp()
        output_path = os.path.join(tmp_dir, "processed.mp4")
        writer = video_proc.create_writer(output_path, fps)

        detections_log = []
        temp_history = []
        max_temp_overall = 0.0
        last_object_id = -1
        telegram_sent = False
        frame_idx = 0

        while True:
            ret, frame = video_proc.read_frame()
            if not ret:
                break

            temp_frame = temp_extractor.extract_from_rgb(frame)
            detections, hot_mask = detector.detect(temp_frame)
            tracked = tracker.update(detections, frame_idx)

            # Check for new confirmed objects
            new_objects = {
                obj_id: info
                for obj_id, info in tracked.items()
                if obj_id > last_object_id
            }

            # Track max temperature
            current_max_temp = 0.0
            if detections:
                current_max_temp = detections[0]["max_temperature"]
                max_temp_overall = max(max_temp_overall, current_max_temp)

            # Record temperature for chart
            if detections:
                chart_temp = detections[0]["max_temperature"]
            else:
                mask = temp_frame[35:-15, 8:]
                chart_temp = float(np.percentile(mask, 95))
            temp_history.append({
                "frame": frame_idx,
                "time": frame_idx / fps if fps > 0 else 0,
                "temp": chart_temp,
            })

            # Render frame with overlays
            output_frame = renderer.render(
                frame, temp_frame, detections, timestamp=video_proc.get_timestamp()
            )

            # Write to output video
            writer.write(output_frame)

            # Log new objects and send alerts
            if new_objects:
                for obj_id, obj_info in new_objects.items():
                    bbox = obj_info["bbox"]
                    timestamp = video_proc.get_timestamp()
                    detections_log.append({
                        "object_id": obj_id,
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "max_temp": obj_info["max_temperature"],
                        "severity": obj_info["severity"],
                        "bbox": bbox,
                        "type": "new_object",
                    })
                    last_object_id = obj_id

                    # Send Telegram alert (first detection only)
                    if not telegram_sent:
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
                                telegram_sent = True

            frame_idx += 1

            # Update progress every 10 frames
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                progress_bar.progress(frame_idx / total_frames)
                status_text.info(f"🔄 Procesando: Frame {frame_idx}/{total_frames}")

        writer.release()
        video_proc.release()

        # Store results in session state
        st.session_state.detections_log = detections_log
        st.session_state.tracked_objects = tracker.get_all_objects()
        st.session_state.max_temp_overall = max_temp_overall
        st.session_state.total_frames = frame_idx
        st.session_state.temp_history = temp_history
        st.session_state.output_video_path = output_path
        st.session_state.processing = False
        st.session_state.processed = True

        st.rerun()

    except Exception as e:
        status_text.error(f"❌ Error: {str(e)}")
        st.exception(e)
        st.session_state.processing = False


# =====================================================================
# PHASE 2: Display results with smooth video playback
# =====================================================================
if st.session_state.processed and not st.session_state.processing:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("### 📹 Video Procesado")
        # Use native st.video for smooth HTML5 playback
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            with open(st.session_state.output_video_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
        else:
            st.warning("Video procesado no disponible.")

    with col2:
        st.markdown("### 📊 Métricas")

        total_objects = len(st.session_state.tracked_objects)
        total_alerts = len(st.session_state.detections_log)
        max_temp = (
            max(d["max_temp"] for d in st.session_state.detections_log)
            if st.session_state.detections_log else 0
        )
        critical_count = sum(
            1 for d in st.session_state.detections_log if d["severity"] == "critical"
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("📊 Total Frames", st.session_state.total_frames)
            st.metric("🔴 Alertas Críticas", critical_count)
        with m2:
            st.metric("🎯 Objetos Únicos", total_objects)
            st.metric("🌡️ Temp. Máxima", f"{max_temp:.1f}°C")

        # Temperature chart
        if len(st.session_state.temp_history) >= 2:
            import matplotlib.pyplot as plt
            import io as _io
            st.markdown("#### 🌡️ Temperatura Máxima")
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
            st.markdown(
                f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;">',
                unsafe_allow_html=True,
            )

        # Alerts
        st.markdown("---")
        st.markdown(
            '<p class="detection-header">🚨 Alertas Detectadas</p>',
            unsafe_allow_html=True,
        )
        if st.session_state.detections_log:
            html = ""
            for alert in reversed(st.session_state.detections_log[-10:]):
                cls = "critical-alert" if alert["severity"] == "critical" else "warning-alert"
                icon = "🔴" if alert["severity"] == "critical" else "🟡"
                bbox = alert.get("bbox", (0, 0, 0, 0))
                html += f"""
                <div class="{cls}">
                    {icon} <strong>OBJETO DETECTADO</strong><br>
                    ID: #{alert["object_id"]} | Frame {alert["frame"]} - {alert["timestamp"]}<br>
                    Temperatura: {alert["max_temp"]:.1f}°C<br>
                    Posición: ({bbox[0]}, {bbox[1]}) | Tamaño: {bbox[2]}x{bbox[3]}px
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No se detectaron anomalías térmicas.")

    # --- Object details ---
    if st.session_state.tracked_objects:
        st.markdown("---")
        st.markdown("### 🔍 Detalles de Objetos Detectados")

        for obj_id, obj_info in st.session_state.tracked_objects.items():
            with st.expander(
                f"📦 Objeto #{obj_id} - {obj_info['max_temperature']:.1f}°C "
                f"({obj_info['severity'].upper()})"
            ):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**Primera detección:** Frame {obj_info['first_seen_frame']}")
                    st.write(f"**Última detección:** Frame {obj_info['last_seen_frame']}")
                with c2:
                    dur = obj_info["last_seen_frame"] - obj_info["first_seen_frame"] + 1
                    st.write(f"**Duración:** {dur} frames")
                    st.write(f"**Detecciones:** {obj_info['total_detections']} veces")
                with c3:
                    st.write(f"**Temp. máxima:** {obj_info['max_temperature']:.1f}°C")
                    st.write(f"**Temp. media:** {obj_info['mean_temperature']:.1f}°C")

    # --- Export ---
    if st.session_state.detections_log or st.session_state.tracked_objects:
        st.markdown("---")
        st.markdown("### 💾 Exportar Resultados")

        exp1, exp2 = st.columns(2)

        with exp1:
            report_data = {
                "metadata": {
                    "date": datetime.now().isoformat(),
                    "video": selected_video,
                    "threshold": temp_threshold,
                    "critical_threshold": critical_threshold,
                    "total_frames": st.session_state.total_frames,
                    "total_objects": total_objects,
                    "total_alerts": total_alerts,
                },
                "tracked_objects": [
                    {"object_id": oid, **info}
                    for oid, info in st.session_state.tracked_objects.items()
                ],
                "alerts": st.session_state.detections_log,
            }
            st.download_button(
                label="📥 Descargar Reporte JSON",
                data=json.dumps(report_data, indent=2),
                file_name=f"thermal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with exp2:
            txt = f"""REPORTE DE MONITOREO TÉRMICO
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Video: {selected_video}

=== RESUMEN ===
Total Frames: {st.session_state.total_frames}
Objetos Únicos: {total_objects}
Alertas: {total_alerts}
Alertas Críticas: {critical_count}
Temp. Máxima: {max_temp:.1f}°C

=== OBJETOS DETECTADOS ===
"""
            for oid, info in st.session_state.tracked_objects.items():
                dur = info["last_seen_frame"] - info["first_seen_frame"] + 1
                txt += f"\nObjeto #{oid}:"
                txt += f"\n  Temp. Máxima: {info['max_temperature']:.1f}°C"
                txt += f"\n  Severidad: {info['severity'].upper()}"
                txt += f"\n  Frames: {info['first_seen_frame']}-{info['last_seen_frame']} ({dur} frames)"
                txt += f"\n  Detecciones: {info['total_detections']}\n"

            st.download_button(
                label="📄 Descargar Resumen TXT",
                data=txt,
                file_name=f"thermal_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

# --- Initial state: show instructions ---
if not st.session_state.processing and not st.session_state.processed:
    st.markdown("---")
    st.markdown("""
    ### 👋 Bienvenido al Sistema de Monitoreo Térmico

    **Instrucciones:**
    1. Selecciona un video térmico en la barra lateral
    2. Ajusta los umbrales de detección si es necesario
    3. Presiona **Iniciar Monitoreo** para procesar el video
    4. El sistema analizará cada frame y generará un video con las detecciones marcadas
    5. Una vez procesado, podrás reproducir el video de forma fluida y revisar las alertas
    """)

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
