mod config;
mod vision;

use anyhow::Result;
use opencv::{
    core::{Point, Scalar},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use vision::YoloDetector;

fn main() -> Result<()> {
    let window_name = "Elegoo Conquerer Vision";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let mut detector = YoloDetector::new()?;

    println!("Connecting to Stream: {}", config::ROBOT_STREAM_URL);
    let mut cam = videoio::VideoCapture::from_file(config::ROBOT_STREAM_URL, videoio::CAP_ANY)?;

    videoio::VideoCapture::is_opened(&cam).expect("Failed to open camera stream");

    let mut frame = Mat::default();

    loop {
        cam.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        let detections = detector.detect(&frame)?;

        for det in detections {
            // Green for Person, Blue for Dog
            let color = if det.label == "person" {
                Scalar::new(0.0, 255.0, 0.0, 0.0)
            } else {
                Scalar::new(255.0, 0.0, 0.0, 0.0)
            };

            // Draw Box
            imgproc::rectangle(&mut frame, det.bbox, color, 2, imgproc::LINE_8, 0)?;

            // Draw Text
            let text = format!("{} {:.0}%", det.label, det.confidence * 100.0);
            imgproc::put_text(
                &mut frame,
                &text,
                Point::new(det.bbox.x, det.bbox.y - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }

        highgui::imshow(window_name, &frame)?;

        if highgui::wait_key(1)? == 113 {
            // 'q' to quit
            break;
        }
    }

    Ok(())
}
