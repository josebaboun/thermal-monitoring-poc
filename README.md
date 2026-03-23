# Thermal Monitoring POC - Sistema de Detección de Polines Sobrecalentados

Sistema de Prueba de Concepto (POC) para detectar polines sobrecalentados en fajas transportadoras mineras mediante análisis de video termográfico.

## 📋 Descripción

Este proyecto procesa videos térmicos capturados con cámara HIKMICRO PocketE para:
- Detectar automáticamente objetos que superan un umbral de temperatura
- Marcar visualmente las detecciones con bounding boxes
- Generar video procesado con overlays informativos
- Crear reportes JSON con todas las detecciones
- (Futuro) Enviar alertas via Telegram/SMS cuando se detecten anomalías

## 🎯 Resultados de la Primera Prueba

### Video Procesado
- **Input**: HM20260224084124.mp4 (3.4MB, 13 segundos, 326 frames)
- **Output**: processed_thermal.mp4 (1.7MB)
- **Detecciones**: 326 (100% de los frames)
- **Temperatura detectada**: 80°C (máxima)
- **Umbral configurado**: 30°C

### Estadísticas
- Resolución: 240x240 píxeles
- FPS: 24.93
- Duración: 13.08 segundos
- Todas las detecciones clasificadas como "critical" (>50°C)

## 🚀 Uso

### Opción 1: Dashboard Interactivo con Streamlit (Recomendado)

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar dashboard
streamlit run dashboard.py
```

El dashboard abrirá en tu navegador (http://localhost:8501) y permite:
- ✅ Ajustar umbral de temperatura en tiempo real
- ✅ Ver procesamiento frame por frame
- ✅ Métricas en vivo
- ✅ Historial de detecciones
- ✅ Descargar reporte JSON

### Opción 2: Procesamiento por línea de comandos

```bash
# Activar entorno virtual
source venv/bin/activate

# Procesar video
python main.py
```

El script procesará el video configurado en `config/config.yaml` y generará:
- Video procesado: `data/output/processed_thermal.mp4`
- Reporte JSON: `data/reports/detections.json`

### 3. Configuración

Edita `config/config.yaml` para ajustar parámetros:

```yaml
detection:
  temperature_threshold: 30.0    # Umbral de alerta (°C)
  critical_threshold: 50.0       # Umbral crítico (°C)
  min_detection_area: 10         # Área mínima en píxeles
```

## 📁 Estructura del Proyecto

```
thermal-monitoring-poc/
├── data/
│   ├── input/              # Videos térmicos de entrada
│   ├── output/             # Videos procesados
│   └── reports/            # Reportes JSON
├── src/
│   ├── core/               # Procesamiento y detección
│   ├── visualization/      # Renderizado de video
│   └── utils/              # Utilidades
├── config/
│   └── config.yaml         # Configuración
└── main.py                 # Script principal
```

## 🔧 Componentes Principales

### 1. Video Processor
- Carga videos térmicos MP4
- Extrae metadata (FPS, resolución)
- Gestiona lectura frame-by-frame

### 2. Temperature Extractor
- Convierte RGB a temperatura estimada
- Mapeo basado en brillo y saturación (HSV)
- Rango configurable (15-80°C por defecto)

### 3. Thermal Detector
- Detección por umbral de temperatura
- Filtrado morfológico para eliminar ruido
- Identificación de regiones conectadas
- Clasificación por severidad (warning/critical)

### 4. Video Renderer
- Bounding boxes rojos para detecciones
- Texto con temperatura máxima
- Timestamp del video
- Contador de detecciones

## 📊 Formato del Reporte

El archivo `detections.json` contiene:

```json
{
  "frame": 0,
  "timestamp": "00:00.040",
  "max_temperature": 80.0,
  "mean_temperature": 62.64,
  "area": 57121.0,
  "severity": "critical",
  "bbox": [0, 0, 240, 240]
}
```

## 🎨 Visualización

El video procesado incluye:
- **Bounding boxes**: Rojos para objetos calientes
- **Temperatura**: Texto con valor máximo detectado
- **Timestamp**: Posición en el video
- **Contador**: Número de detecciones activas

## ⚙️ Dependencias

- opencv-python - Procesamiento de video
- numpy - Operaciones matriciales
- matplotlib - Colormaps térmicos
- pillow - Procesamiento de imágenes
- pyyaml - Configuración
- loguru - Logging

## 🔮 Próximos Pasos

1. **Sistema de Alertas**
   - Implementar notificaciones Telegram
   - Integrar SMS vía Twilio
   - Webhook para Discord/Slack

2. **Calibración**
   - Ajustar mapeo RGB → Temperatura
   - Probar con datos radiométricos reales
   - Validar precisión de temperaturas

3. **Mejoras de Detección**
   - Tracking de objetos entre frames
   - Reducción de falsos positivos
   - Análisis de tendencias temporales

4. **Dashboard Interactivo**
   - Interfaz Streamlit
   - Control de parámetros en tiempo real
   - Visualización de métricas

5. **Optimización**
   - Procesamiento en paralelo
   - Compresión de video de salida
   - Batch processing de múltiples videos

## 📝 Notas Técnicas

### Mapeo de Temperatura
El sistema usa aproximación RGB→Temperatura basada en:
- Canal V (brillo) de HSV como proxy principal
- Saturación inversa para detectar píxeles blancos (muy calientes)
- Rango calibrado: 15°C (ambiente) a 80°C (máximo)

**Limitación**: Sin datos radiométricos reales, la temperatura es estimada. Para precisión absoluta, se requiere exportar TIFF 16-bit con metadatos de calibración desde HIKMICRO Studio.

### Detección
- Umbral configurable (actualmente 30°C)
- Filtros morfológicos (closing + opening) para ruido
- Área mínima: 10 píxeles
- Clasificación: warning (30-50°C) / critical (>50°C)

## 👤 Autor

José - Proyecto de monitoreo térmico para fajas transportadoras mineras

## 📄 Licencia

Proyecto POC interno
