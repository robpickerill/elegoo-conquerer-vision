pub const ROBOT_STREAM_URL: &str = "http://192.168.4.1:81/stream";

// Files
pub const CONFIG_FILE: &str = "yolov4-tiny.cfg";
pub const WEIGHTS_FILE: &str = "yolov4-tiny.weights";
pub const CLASSES_FILE: &str = "coco.names";

// AI Thresholds
pub const CONF_THRESHOLD: f32 = 0.5; // Minimum confidence to detect
pub const NMS_THRESHOLD: f32 = 0.4; // Non-Maximum Suppression
pub const INPUT_SIZE: i32 = 416; // YOLOv4-tiny resolution
