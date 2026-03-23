"""Telegram alert notifier for thermal detections."""
import io
import cv2
import numpy as np
import requests
from typing import Optional


class TelegramNotifier:
    """Sends thermal detection alerts to a Telegram chat."""

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._validated = False

    def validate(self) -> bool:
        """Check that the bot token is valid."""
        try:
            url = self.API_URL.format(token=self.bot_token, method="getMe")
            resp = requests.get(url, timeout=10)
            self._validated = resp.ok
            return self._validated
        except requests.RequestException:
            return False

    def send_alert(
        self,
        frame_bgr: np.ndarray,
        object_id: int,
        temperature: float,
        severity: str,
        bbox: tuple,
        timestamp: str,
        frame_number: int,
    ) -> bool:
        """
        Send a detection alert with the annotated frame image.

        Returns True if sent successfully.
        """
        # Build message text
        if severity == "critical":
            header = "🔴 ALERTA CRÍTICA - Anomalía Térmica"
        else:
            header = "🟡 ADVERTENCIA - Anomalía Térmica"

        caption = (
            f"{header}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌡️ Temperatura: {temperature:.1f}°C\n"
            f"⚠️ Severidad: {'CRÍTICA' if severity == 'critical' else 'ADVERTENCIA'}\n"
            f"📍 Coordenadas: x={bbox[0]}, y={bbox[1]}\n"
            f"📐 Tamaño detección: {bbox[2]}x{bbox[3]}px\n"
            f"⏱️ Tiempo: {timestamp}\n"
            f"🎞️ Frame: {frame_number}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Sistema de Monitoreo Térmico POC"
        )

        # Encode frame as JPEG in memory
        success, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return self._send_text(caption)

        return self._send_photo(buf.tobytes(), caption)

    def send_summary(self, total_frames: int, total_objects: int,
                     total_alerts: int, max_temp: float) -> bool:
        """Send a processing summary message."""
        text = (
            "📊 *Resumen de Monitoreo*\n\n"
            f"🎞️ Frames procesados: {total_frames}\n"
            f"📦 Objetos detectados: {total_objects}\n"
            f"🚨 Alertas generadas: {total_alerts}\n"
            f"🌡️ Temp. máxima: {max_temp:.1f}°C"
        )
        return self._send_text(text, parse_mode="Markdown")

    def _send_photo(self, image_bytes: bytes, caption: str) -> bool:
        """Send a photo with caption."""
        try:
            url = self.API_URL.format(token=self.bot_token, method="sendPhoto")
            resp = requests.post(
                url,
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": ("alert.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                timeout=30,
            )
            return resp.ok
        except requests.RequestException:
            return False

    def _send_text(self, text: str, parse_mode: str = None) -> bool:
        """Send a text message."""
        try:
            url = self.API_URL.format(token=self.bot_token, method="sendMessage")
            data = {"chat_id": self.chat_id, "text": text}
            if parse_mode:
                data["parse_mode"] = parse_mode
            resp = requests.post(url, data=data, timeout=15)
            return resp.ok
        except requests.RequestException:
            return False
