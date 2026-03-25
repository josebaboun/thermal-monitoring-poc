"""Main processing script for thermal monitoring POC."""
import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / '.env')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.video_processor import VideoProcessor
from core.temperature_extractor import TemperatureExtractor
from core.thermal_detector import ThermalDetector
from core.object_tracker import ThermalObjectTracker
from visualization.video_renderer import VideoRenderer
from alerts.telegram_notifier import TelegramNotifier
from utils.config import Config
from utils.logger import log


def main():
    """Main processing pipeline."""
    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    config = Config.load(str(config_path))

    log.info("🔥 Thermal Monitoring POC Started")
    log.info(f"System: {config.get('system.name')} v{config.get('system.version')}")

    # Initialize components
    log.info("Initializing components...")
    video_proc = VideoProcessor(config.video)
    temp_extractor = TemperatureExtractor(config.thermal)
    detector = ThermalDetector(config.detection)
    tracker = ThermalObjectTracker(
        max_disappeared=config.get('tracking.max_disappeared', 75),
        max_distance=config.get('tracking.max_distance', 150.0),
        min_confirm_frames=config.get('tracking.min_confirm_frames', 5),
        critical_threshold=config.get('detection.critical_threshold', 60.0)
    )
    renderer = VideoRenderer(config.visualization)

    # Initialize Telegram alerts (optional)
    telegram = None
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    if bot_token and chat_id:
        telegram = TelegramNotifier(bot_token, chat_id)
        if telegram.validate():
            log.info("✅ Telegram alerts enabled")
        else:
            log.warning("⚠️ Telegram bot token invalid, alerts disabled")
            telegram = None
    else:
        log.info("Telegram alerts disabled (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")

    # Load video
    input_path = Path(__file__).parent / config.video['input_path']
    log.info(f"Loading video: {input_path}")
    video_proc.load(str(input_path))

    # Log video info
    video_info = video_proc.get_video_info()
    log.info(f"Video Info:")
    log.info(f"  Resolution: {video_info['width']}x{video_info['height']}")
    log.info(f"  FPS: {video_info['fps']:.2f}")
    log.info(f"  Total Frames: {video_info['total_frames']}")
    log.info(f"  Duration: {video_info['duration_seconds']:.2f}s")

    # Create output writer
    output_path = Path(__file__).parent / config.video['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Output video: {output_path}")

    output_writer = video_proc.create_writer(str(output_path))

    # Processing statistics
    frame_count = 0
    detection_log = []
    last_object_id = -1
    telegram_sent = False

    log.info("Starting frame processing...")
    log.info(f"Temperature threshold: {config.detection['temperature_threshold']}°C")

    # Process frames
    while True:
        ret, frame = video_proc.read_frame()
        if not ret:
            break

        # Extract temperature from frame
        temp_frame = temp_extractor.extract_from_rgb(frame)

        # Detect anomalies
        detections, hot_mask = detector.detect(temp_frame)

        # Update object tracker
        tracked = tracker.update(detections, frame_count)

        # Render frame with overlays
        output_frame = renderer.render(
            frame,
            temp_frame,
            detections,
            timestamp=video_proc.get_timestamp()
        )

        # Check for NEW objects (not just detections)
        new_objects = {
            obj_id: info
            for obj_id, info in tracked.items()
            if obj_id > last_object_id
        }

        # Log NEW objects only
        if new_objects:
            timestamp = video_proc.get_timestamp()
            for obj_id, obj_info in new_objects.items():
                bbox = obj_info['bbox']
                log.warning(
                    f"Frame {frame_count} ({timestamp}): "
                    f"NEW OBJECT #{obj_id} detected - "
                    f"Temp: {obj_info['max_temperature']:.1f}°C - "
                    f"Severity: {obj_info['severity']} - "
                    f"Position: ({bbox[0]}, {bbox[1]}) Size: {bbox[2]}x{bbox[3]}"
                )

                detection_log.append({
                    'object_id': obj_id,
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'max_temperature': obj_info['max_temperature'],
                    'mean_temperature': obj_info['mean_temperature'],
                    'severity': obj_info['severity'],
                    'bbox': bbox,
                    'type': 'new_object'
                })

                # Send Telegram alert (only first detection per video)
                if telegram and not telegram_sent:
                    sent = telegram.send_alert(
                        frame_bgr=output_frame,
                        object_id=obj_id,
                        temperature=obj_info['max_temperature'],
                        severity=obj_info['severity'],
                        bbox=bbox,
                        timestamp=timestamp,
                        frame_number=frame_count,
                    )
                    if sent:
                        log.info(f"  → Telegram alert sent for object #{obj_id}")
                        telegram_sent = True

                last_object_id = obj_id

        # Write to output
        output_writer.write(output_frame)

        frame_count += 1

        # Progress update every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / video_info['total_frames']) * 100
            log.info(f"Progress: {frame_count}/{video_info['total_frames']} ({progress:.1f}%)")

    # Cleanup
    video_proc.release()
    output_writer.release()

    # Final statistics
    active_count, total_count = tracker.get_object_count()
    all_objects = tracker.get_all_objects()

    log.success("✅ Processing completed!")
    log.info(f"Frames processed: {frame_count}")
    log.info(f"Unique objects detected: {total_count}")
    log.info(f"Total alerts generated: {len(detection_log)}")

    # Show object details
    if all_objects:
        log.info("\n=== DETECTED OBJECTS ===")
        for obj_id, obj_info in all_objects.items():
            duration = obj_info['last_seen_frame'] - obj_info['first_seen_frame'] + 1
            log.info(
                f"Object #{obj_id}: {obj_info['max_temperature']:.1f}°C - "
                f"Frames {obj_info['first_seen_frame']}-{obj_info['last_seen_frame']} "
                f"({duration} frames, {obj_info['total_detections']} detections) - "
                f"Severity: {obj_info['severity']}"
            )

    # Save detection report with tracked objects
    report_path = Path(__file__).parent / 'data' / 'reports' / 'detections.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = {
        'metadata': {
            'total_frames': frame_count,
            'total_objects': total_count,
            'total_alerts': len(detection_log),
            'threshold': config.detection['temperature_threshold'],
            'critical_threshold': config.detection['critical_threshold']
        },
        'tracked_objects': [
            {
                'object_id': obj_id,
                **obj_info
            }
            for obj_id, obj_info in all_objects.items()
        ],
        'alerts': detection_log
    }

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    log.info(f"Detection report saved: {report_path}")
    log.info(f"Unique objects: {total_count}")
    log.info(f"Total alerts: {len(detection_log)}")

    # Send Telegram summary
    if telegram:
        max_temp = max((d['max_temperature'] for d in detection_log), default=0)
        telegram.send_summary(frame_count, total_count, len(detection_log), max_temp)
        log.info("📊 Telegram summary sent")

    log.info(f"Output video saved: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
