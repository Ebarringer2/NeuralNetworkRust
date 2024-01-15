pub mod node {

    use ndarray::{Array1, ArrayView1};
    use std::fmt;

    /// calculates the dot product of two ndarray type objects
    pub fn dot_product(arr1: ArrayView1<f64>, arr2: ArrayView1<f64>) -> f64 {
        arr1.dot(&arr2)
    }

    /// Shortcut for the dot product value used in node activation equations
    pub fn create_z(features: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>, weights: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>, bias: f64) -> f64 {
        dot_product(features.view(), weights.view()) + bias
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
        /// Sigmoid Actualization
        pub fn sigmoid_actualize(&self, features: Vec<f64>) -> f64 {
            println!("Sigmoid activating");
            let features_arr: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>> = Array1::from_vec(features);
            let weights_arr: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<_>, ndarray::prelude::Dim<[usize; 1]>> = Array1::from_vec(self.weights.clone());
            println!("calculating dot product of features and weights");
            //println!("Features: {}", features_arr.view());
            //println!("Weights: {}", weights_arr.view());
            let z: f64 = dot_product(features_arr.view(), weights_arr.view()) + self.bias;
            println!("done");
            let exp_z: f64 = (-z).exp();
            let output: f64 = 1.0 / (1.0 + exp_z);
            println!("Activation output: {}", output);
            output
        }

        pub fn reLU_activate(&self, features: Vec<f64>) -> f64 {
            println!("ReLU activating");
            let z: f64 = create_z(Array1::from_vec(features), Array1::from_vec(self.weights.clone()), self.bias);
            let output: f64 = f64::max(0.0, z);
            println!("Activation output: {}", output);
            output
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