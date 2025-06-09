// Chart component for UNIA OS UI
use alloc::string::String;
use alloc::vec::Vec;

pub enum ChartType {
    Line,
    Bar,
    Pie,
}

pub struct Chart {
    title: String,
    chart_type: ChartType,
    data: Vec<f32>,
    labels: Vec<String>,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Chart {
    pub fn new(title: String, chart_type: ChartType, x: usize, y: usize, width: usize, height: usize) -> Self {
        Chart {
            title,
            chart_type,
            data: Vec::new(),
            labels: Vec::new(),
            x,
            y,
            width,
            height,
        }
    }

    pub fn add_data_point(&mut self, value: f32, label: String) {
        self.data.push(value);
        self.labels.push(label);
    }

    pub fn get_title(&self) -> &str {
        &self.title
    }

    pub fn get_data(&self) -> &[f32] {
        &self.data
    }

    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }

    pub fn get_position(&self) -> (usize, usize) {
        (self.x, self.y)
    }

    pub fn get_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}
