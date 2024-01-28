pub mod node {

    use ndarray::{Array1, ArrayView1};
    use std::f64::consts::E;

    pub struct Node {
        weights: Vec<f64>,
        num_weights: usize,
        b: f64,
    }

    impl Clone for Node {
        fn clone(&self) -> Self {
            *self
        }
    }


    impl Node {
        pub fn new(weights: Vec<f64>, b: f64) -> Self {
            let num_weights = weights.len();
            Node { weights, num_weights, b }
        }

        // Range 0 - 1
        pub fn sigmoid_actualize(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            1.0 / (1.0 + (-z).exp())
        }

        // Range 0 - infinity
        pub fn relu_actualize(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            z.max(0.0)
        }

        // Range -1 - 1
        pub fn tanh_activation(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            let a = z.exp();
            let b = (-z).exp();
            (a - b) / (a + b)
        }

        // Enables back propagation for ReLu
        // Enables negative signed inputs, which means
        // that the gradient on the left side of the activation graph
        // is non-zero, enabling back propagation
        pub fn leaky_relu_activation(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            z.max(0.1 * z)
        }

        // Parametric Relu can be used when leaky Relu doesn't solve the
        // zero gradient problem for Relu activation
        // Creates problems because the solution is to use a slope value
        // for negative inputs, but there can be difficulty finding the
        // correct slope value
        pub fn parametric_relu_activation(&self, features: &Array1<f64>, a: f64) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            z.max(a * z)
        }

        // Uses log curve to define negative inputs
        // a helps define the log curve
        pub fn elu_activation(&self, features: &Array1<f64>, a: f64) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            if z >= 0.0 {
                z
            } else {
                a * ((-z).exp() - 1.0)
            }
        }

        // Useful for multi-class classification problems
        pub fn softmax(features: ArrayView1<f64>) -> f64 {
            assert!(!features.is_empty(), "Softmax input array cannot be empty");
            let exp_values: Array1<f64> = features.mapv(|x| E.powf(x));
            let sum_exp: f64 = exp_values.sum();
            let softmax_values: Array1<f64> = exp_values / sum_exp;
            let result: f64 = softmax_values.sum();
            result
        }


        // Consistently outperforms or performs at the same level as Relu activation
        // Is literally just z * sigmoid_actualize(z)
        pub fn swish(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            let sigmoid = 1.0 / (1.0 + (-z).exp());
            z * sigmoid
        }

        // GELU implementation
        pub fn gelu_activation(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            let coefficient = (2.0 / std::f64::consts::PI).sqrt();
            0.5 * z * (1.0 + (coefficient * (z + 0.044715 * z.powi(3))).tanh())
        }

        pub fn set_weights(&mut self, weights: Vec<f64>, b: f64) {
            self.weights = weights;
            self.b = b;
            self.num_weights = weights.len();
        }

        pub fn get_weights(&self) -> &Vec<f64> {
            &self.weights
        }

        pub fn get_bias(&self) -> f64 {
            self.b
        }
    }
}