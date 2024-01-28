pub mod node {

    use ndarray::Array1;

    struct Node {
        weights: Vec<f64>,
        num_weights: usize,
        b: f64,
    }

    impl Node {
        fn new(weights: Vec<f64>, b: f64) -> Self {
            let num_weights = weights.len();
            Node { weights, num_weights, b }
        }

        // Range 0 - 1
        fn sigmoid_actualize(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            1.0 / (1.0 + (-z).exp())
        }

        // Range 0 - infinity
        fn relu_actualize(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            z.max(0.0)
        }

        // Range -1 - 1
        fn tanh_activation(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            let a = z.exp();
            let b = (-z).exp();
            (a - b) / (a + b)
        }

        // Enables back propagation for ReLu
        // Enables negative signed inputs, which means
        // that the gradient on the left side of the activation graph
        // is non-zero, enabling back propagation
        fn leaky_relu_activation(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            z.max(0.1 * z)
        }

        // Parametric Relu can be used when leaky Relu doesn't solve the
        // zero gradient problem for Relu activation
        // Creates problems because the solution is to use a slope value
        // for negative inputs, but there can be difficulty finding the
        // correct slope value
        fn parametric_relu_activation(&self, features: &Array1<f64>, a: f64) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            z.max(a * z)
        }

        // Uses log curve to define negative inputs
        // a helps define the log curve
        fn elu_activation(&self, features: &Array1<f64>, a: f64) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            if z >= 0.0 {
                z
            } else {
                a * ((-z).exp() - 1.0)
            }
        }

        // Useful for multi-class classification problems
        fn softmax_activation(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            let max_x = features.max().unwrap();
            let e_x = (z - max_x).exp();
            e_x / e_x.sum()
        }

        // Consistently outperforms or performs at the same level as Relu activation
        // Is literally just z * sigmoid_actualize(z)
        fn swish(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            let sigmoid = 1.0 / (1.0 + (-z).exp());
            z * sigmoid
        }

        // GELU implementation
        fn gelu_activation(&self, features: &Array1<f64>) -> f64 {
            let z = features.dot(&Array1::from(self.weights.clone())) + self.b;
            let coefficient = (2.0 / std::f64::consts::PI).sqrt();
            0.5 * z * (1.0 + (coefficient * (z + 0.044715 * z.powi(3))).tanh())
        }

        fn set_weights(&mut self, weights: Vec<f64>, b: f64) {
            self.weights = weights;
            self.b = b;
            self.num_weights = weights.len();
        }

        fn get_weights(&self) -> &Vec<f64> {
            &self.weights
        }

        fn get_bias(&self) -> f64 {
            self.b
        }
    }
}