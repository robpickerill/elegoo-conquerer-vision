use anyhow::{Context, Result};
use opencv::{
    core::{self, Rect, Scalar, Size, Vector},
    dnn::{self, Net, NetTrait},
    prelude::*,
};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::config;

/// Represents a single detected object
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub bbox: Rect,
    pub class_id: i32,
}

pub struct YoloDetector {
    net: Net,
    classes: Vec<String>,
    out_names: Vector<String>,
    person_id: i32,
    dog_id: i32,
}

impl YoloDetector {
    /// Initialize the Neural Network
    pub fn new() -> Result<Self> {
        let classes = Self::load_classes(config::CLASSES_FILE)?;
        let person_id = classes
            .iter()
            .position(|r| r == "person")
            .ok_or_else(|| anyhow::anyhow!("Class 'person' not found"))?
            as i32;
        let dog_id = classes
            .iter()
            .position(|r| r == "dog")
            .ok_or_else(|| anyhow::anyhow!("Class 'dog' not found"))? as i32;

        let mut net = dnn::read_net_from_darknet(config::CONFIG_FILE, config::WEIGHTS_FILE)?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

        let out_layers = net.get_unconnected_out_layers()?;
        let layer_names = net.get_layer_names()?;
        let mut out_names = Vector::<String>::new();

        for i in out_layers.iter() {
            let name = layer_names.get(i as usize - 1)?;
            out_names.push(name.as_str());
        }

        Ok(Self {
            net,
            classes,
            out_names,
            person_id,
            dog_id,
        })
    }

    /// Process a frame and return specific detections
    pub fn detect(&mut self, frame: &Mat) -> Result<Vec<Detection>> {
        let blob = dnn::blob_from_image(
            frame,
            1.0 / 255.0,
            Size::new(config::INPUT_SIZE, config::INPUT_SIZE),
            Scalar::default(),
            true,
            false,
            core::CV_32F,
        )?;

        self.net.set_input(&blob, "", 1.0, Scalar::default())?;

        let mut outputs = Vector::<Mat>::new();
        self.net.forward(&mut outputs, &self.out_names)?;

        self.process_outputs(&outputs, frame)
    }

    /// Internal helper to parse neural network output
    fn process_outputs(&self, outputs: &Vector<Mat>, frame: &Mat) -> Result<Vec<Detection>> {
        let mut boxes = Vector::<Rect>::new();
        let mut confidences = Vector::<f32>::new();
        let mut class_ids = Vector::<i32>::new();

        for output in outputs.iter() {
            for row in 0..output.rows() {
                let row_data = output.at_row::<f32>(row)?;
                let confidence = row_data[4];

                if confidence > config::CONF_THRESHOLD {
                    let (class_id, max_score) = self.get_best_class(&row_data[5..]);

                    if max_score > config::CONF_THRESHOLD {
                        // FILTER: Only keep Humans or Dogs
                        if class_id == self.person_id || class_id == self.dog_id {
                            let bbox = self.calculate_bbox(row_data, frame.cols(), frame.rows());
                            boxes.push(bbox);
                            confidences.push(confidence);
                            class_ids.push(class_id);
                        }
                    }
                }
            }
        }

        // Apply Non-Maximum Suppression (remove duplicates)
        let mut indices = Vector::<i32>::new();
        dnn::nms_boxes(
            &boxes,
            &confidences,
            config::CONF_THRESHOLD,
            config::NMS_THRESHOLD,
            &mut indices,
            1.0,
            0,
        )?;

        // Compile final results
        let mut results = Vec::new();
        for i in indices.iter() {
            let i = i as usize;
            let cls_id = class_ids.get(i)?;

            results.push(Detection {
                label: self.classes[cls_id as usize].clone(),
                confidence: confidences.get(i)?,
                bbox: boxes.get(i)?,
                class_id: cls_id,
            });
        }
        Ok(results)
    }

    fn get_best_class(&self, scores: &[f32]) -> (i32, f32) {
        let mut max_score = 0.0;
        let mut id = -1;
        for (i, &score) in scores.iter().enumerate() {
            if score > max_score {
                max_score = score;
                id = i as i32;
            }
        }
        (id, max_score)
    }

    fn calculate_bbox(&self, row_data: &[f32], frame_w: i32, frame_h: i32) -> Rect {
        let cx = (row_data[0] * frame_w as f32) as i32;
        let cy = (row_data[1] * frame_h as f32) as i32;
        let w = (row_data[2] * frame_w as f32) as i32;
        let h = (row_data[3] * frame_h as f32) as i32;
        Rect::new(cx - w / 2, cy - h / 2, w, h)
    }

    fn load_classes(path: &str) -> Result<Vec<String>> {
        let file = File::open(path).context(format!("Failed to open {}", path))?;
        let reader = BufReader::new(file);
        reader.lines().map(|l| Ok(l?)).collect()
    }
}
