pub mod image_tensor {
    
    extern crate image;
    
    use ndarray::{Array3, Axis};
    use std::path::Path;

    pub struct ImageProcessor;

    impl ImageProcessor {
        
        /// Reads an inputted JPEG image and returns a tensor representation of it
        pub fn read_to_tensor(file_path: &str) -> Option<Array3<f64>> {
            if let Ok(img) = image::open(file_path) {
                let grayscale_arr = ImageProcessor::convert_to_grayscale_array(&img);
                let normalized_array = ImageProcessor::normalize_array(grayscale_arr);
                return Some(normalized_array);
            }
            None
        }

        fn convert_to_grayscale_array(img: &image::DynamicImage) -> Array3<f64> {
            let grayscale_img = img.to_luma8();
            let img_array: Vec<f64> = grayscale_img
                .pixels()
                .flat_map(|pixel| vec![pixel.0[0] as f64 / 255.0])
                .collect();
            let (height, width) = grayscale_img.dimensions();
            let tensor_array = Array3::from_shape_vec((height as usize, width as usize, 1), img_array)
                .expect("Failed to create a tensor array");
            tensor_array
        }

        fn normalize_array(array: Array3<f64>) -> Array3<f64> {
            let max_value = array.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_value = array.iter().cloned().fold(f64::INFINITY, f64::min);
            (array - min_value) / (max_value - min_value)
        }

    }

}
