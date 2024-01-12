pub mod node {

    use ndarray::{Array1, ArrayView1};
    use std::fmt;

    /// calculates the dot product of two ndarray type objects
    pub fn dot_product(arr1: ArrayView1<f64>, arr2: ArrayView1<f64>) -> f64 {
        arr1.dot(&arr2)
    }

    #[derive(Clone)]
    pub struct Node {
        pub weights: Vec<f64>,
        pub bias: f64
    }

    impl fmt::Display for Node {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Weights: {:?}, bias: {}", self.weights, self.bias)
        }
    }

    impl Node {
        /// Creates a new Node Object with weights and a bias
        pub fn new(weights: Vec<f64>, bias: f64) -> Self {
            Node {
                weights,
                bias
            }
        }
        pub fn sigmoid_actualize(&self, features: Vec<f64>) -> f64 {
            let features_arr: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>> = Array1::from_vec(features);
            let weights_arr: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<_>, ndarray::prelude::Dim<[usize; 1]>> = Array1::from_vec(self.weights.clone());
            println!("calculating dot product of features and weights");
            println!("Features: {}", features_arr.view());
            println!("Weights: {}", weights_arr.view());
            let z: f64 = dot_product(features_arr.view(), weights_arr.view()) + self.bias;
            println!("done");
            let exp_z: f64 = (-z).exp();
            1.0 / (1.0 + exp_z)
        }
        pub fn set_weights(&mut self, weights: Vec<f64>, bias: f64) {
            self.weights = weights;
            self.bias = bias;
        }
        pub fn get_weights(&self) -> (Vec<f64>, f64) {
            (self.weights.clone(), self.bias)
        }
        pub fn str(&self) {
            println!("{}", &self)
        }
    }
}