"""Calibration tool for thermal temperature mapping."""
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.video_processor import VideoProcessor
from core.temperature_extractor import TemperatureExtractor
from utils.config import Config

def analyze_video_thermal_range(video_path):
    """Analyze the thermal range in a video to help calibration."""

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Analyze color distribution
    print("\n" + "="*60)
    print("ANÁLISIS DE VIDEO TÉRMICO")
    print("="*60)
    print(f"\nVideo: {video_path.name}")
    print(f"Resolución: {frame.shape[1]}x{frame.shape[0]}")

    print("\n--- Rango HSV ---")
    print(f"Hue (H):        Min={h.min()}, Max={h.max()}, Mean={h.mean():.1f}")
    print(f"Saturation (S): Min={s.min()}, Max={s.max()}, Mean={s.mean():.1f}")
    print(f"Value (V):      Min={v.min()}, Max={v.max()}, Mean={v.mean():.1f}")

    # Analyze BGR channels
    b, g, r = cv2.split(frame)
    print("\n--- Rango RGB ---")
    print(f"Red (R):   Min={r.min()}, Max={r.max()}, Mean={r.mean():.1f}")
    print(f"Green (G): Min={g.min()}, Max={g.max()}, Mean={g.mean():.1f}")
    print(f"Blue (B):  Min={b.min()}, Max={b.max()}, Mean={b.mean():.1f}")

    # Current temperature extraction
    config = Config.load(str(Path(__file__).parent / 'config' / 'config.yaml'))
    temp_extractor = TemperatureExtractor(config.thermal)
    temp_frame = temp_extractor.extract_from_rgb(frame)

    print("\n--- Temperatura Estimada (Algoritmo Actual) ---")
    print(f"Min: {temp_frame.min():.1f}°C")
    print(f"Max: {temp_frame.max():.1f}°C")
    print(f"Mean: {temp_frame.mean():.1f}°C")
    print(f"Median: {np.median(temp_frame):.1f}°C")

    # Histogram analysis
    print("\n--- Distribución de Temperatura ---")
    hist, bins = np.histogram(temp_frame.flatten(), bins=10)
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / hist.max() * 50)
        print(f"{bins[i]:5.1f}°C - {bins[i+1]:5.1f}°C: {bar} ({hist[i]:6d} pixels)")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Análisis de Calibración Térmica', fontsize=16)

    # Original frame (BGR to RGB)
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Frame Original (Thermal Rainbow)')
    axes[0, 0].axis('off')

    # HSV channels
    axes[0, 1].imshow(h, cmap='hsv')
    axes[0, 1].set_title(f'Hue (H)\nMin={h.min()}, Max={h.max()}')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(s, cmap='gray')
    axes[0, 2].set_title(f'Saturation (S)\nMin={s.min()}, Max={s.max()}')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(v, cmap='gray')
    axes[1, 0].set_title(f'Value (V)\nMin={v.min()}, Max={v.max()}')
    axes[1, 0].axis('off')

    # Temperature estimation
    axes[1, 1].imshow(temp_frame, cmap='inferno')
    axes[1, 1].set_title(f'Temperatura Estimada\nMin={temp_frame.min():.1f}°C, Max={temp_frame.max():.1f}°C')
    axes[1, 1].axis('off')

    # Temperature histogram
    axes[1, 2].hist(temp_frame.flatten(), bins=50, color='red', alpha=0.7)
    axes[1, 2].set_xlabel('Temperatura (°C)')
    axes[1, 2].set_ylabel('Frecuencia')
    axes[1, 2].set_title('Distribución de Temperatura')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'data' / 'reports' / 'calibration_analysis.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Análisis guardado en: {output_path}")

    plt.show()

    cap.release()

    # Recommendations
    print("\n" + "="*60)
    print("RECOMENDACIONES DE CALIBRACIÓN")
    print("="*60)

    # Expected temperature for indoor scene with a dog
    expected_ambient = 20  # °C
    expected_dog = 38      # °C (dog body temperature)

    current_range = temp_frame.max() - temp_frame.min()

    print(f"\nEscenario esperado (perro en departamento):")
    print(f"  - Temperatura ambiente: ~{expected_ambient}°C")
    print(f"  - Temperatura del perro: ~{expected_dog}°C")
    print(f"  - Rango esperado: ~{expected_dog - expected_ambient}°C")

    print(f"\nDetectado actualmente:")
    print(f"  - Rango: {current_range:.1f}°C")
    print(f"  - Min: {temp_frame.min():.1f}°C")
    print(f"  - Max: {temp_frame.max():.1f}°C")

    if temp_frame.min() > 25:
        print("\n⚠️  PROBLEMA: La temperatura mínima es muy alta!")
        print("   Esto indica que el mapeo está sobreestimando temperaturas bajas.")
        print("   El fondo (paredes, piso) debería estar cerca de 20°C.")

    if current_range < 10:
        print("\n⚠️  PROBLEMA: El rango de temperatura es muy estrecho!")
        print("   El algoritmo no está capturando bien la diferencia térmica.")

    # Suggest new calibration based on HSV analysis
    print("\n--- Sugerencias de Calibración ---")

    # For rainbow colormap in thermal cameras:
    # Blue/Purple (low H, high S) = Cold
    # Red/Orange/Yellow (high H or low H, high S) = Hot
    # White (low S, high V) = Very hot

    # Analyze what values correspond to likely "cold" areas
    mask_dark = v < 100  # Dark areas are likely cold
    if mask_dark.any():
        avg_temp_dark = temp_frame[mask_dark].mean()
        print(f"\nÁreas oscuras (V<100): {avg_temp_dark:.1f}°C promedio")
        print(f"  → Deberían estar cerca de {expected_ambient}°C")

    mask_bright = v > 200  # Bright areas might be hot
    if mask_bright.any():
        avg_temp_bright = temp_frame[mask_bright].mean()
        print(f"\nÁreas brillantes (V>200): {avg_temp_bright:.1f}°C promedio")
        print(f"  → Podrían ser zonas calientes (~{expected_dog}°C)")

    print("\n💡 ACCIÓN RECOMENDADA:")
    print("   1. Ajusta 'temp_range_min' en config.yaml al valor mínimo esperado")
    print("   2. Ajusta 'temp_range_max' al valor máximo esperado")
    print("   3. O mejor: usa el nuevo algoritmo de calibración automática")


if __name__ == "__main__":
    # Find video
    video_files = list(Path(__file__).parent.glob('data/input/*.mp4'))

    if not video_files:
        print("Error: No se encontraron videos en data/input/")
        sys.exit(1)

    if len(video_files) > 1:
        print("Videos disponibles:")
        for i, v in enumerate(video_files):
            print(f"  {i+1}. {v.name}")
        choice = int(input("Selecciona número de video: ")) - 1
        video_path = video_files[choice]
    else:
        video_path = video_files[0]

    analyze_video_thermal_range(video_path)
