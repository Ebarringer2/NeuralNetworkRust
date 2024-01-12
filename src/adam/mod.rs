pub mod adam {

    /// Adam optimization algorithm implementation
    /// Src: https://arxiv.org/abs/1412.6980

    use crate::gradient_descent::obj::obj::GradientDescent;
    use crate::nn::nn::nn::NeuralNetwork;
    //use std::borrow::Borrow;
    //use crate::batch;
    //use crate::batch_vectorized;
    //use crate::stochastic_vect;
    
    pub struct Adam {
        nn: NeuralNetwork,
        gd: GradientDescent,
        pub stepsize: f64,
        pub beta_1: f64,
        pub beta_2: f64,
        pub epsilon: f64,
        pub m_b: f64,
        pub v_b: f64
    }

    impl Adam {

        /// Creates a new Adam optimizer object. If default is passed as true,
        /// then the base coefficients for the stepsize and exponential decay rates are set to
        /// default, being alpha = 0.0001, beta 1 = 0.9, beta 2 = 0.999, and epsilon = 1e-8
        /// When not using default coefficients, the new function can intake values for 
        /// each of these parameters
        ///
        /// ### Parameters
        /// 
        /// a: stepsize
        /// 
        /// beta_1 & beta_2: exponential decay rates for the moment estimates
        /// 
        /// epsilon: small constant used to avoid division by zero when calculating RMS of the gradients
        pub fn new(nn: NeuralNetwork, gd: GradientDescent, default: bool, stepsize: f64, beta_1: f64, beta_2: f64, epsilon: f64) -> Adam {
            if default {
                let stepsize: f64 = 0.0001;
                let beta_1: f64 = 0.9;
                let beta_2: f64 = 0.999;
                let epsilon: f64 = 1e-8;
                println!("Loading default Adam parameters: alpha: {}, beta 1: {}, beta 2: {}, epsilon: {}", stepsize, beta_1, beta_2, epsilon);
                Adam {
                    nn,
                    gd,
                    stepsize,
                    beta_1,
                    beta_2,
                    epsilon,
                    m_b: 0.0,
                    v_b: 0.0
                }
            } else {
                println!("Using specified Adam parameters: alpha: {}, beta 1: {}, beta 2: {}, epsilon: {}", stepsize, beta_1, beta_2, epsilon);
                Adam {
                    nn,
                    gd,
                    stepsize,
                    beta_1,
                    beta_2,
                    epsilon,
                    m_b: 0.0,
                    v_b: 0.0
                }
            }
        }

        pub fn optimize(&mut self, epochs: usize) {
            let m: f64 = self.gd.x_train.len() as f64;
            let mut gd = self.gd.clone();
            for epoch in 0..epochs {
                for (predictor, output) in self.gd.train_data() {
                    let error = self.gd.h(predictor.clone()) - output;
                    let gradients: Vec<f64> = predictor
                        .iter()
                        .enumerate()
                        .map(|(j, &x)| (self.gd.learning_rate / m) * error * x)
                        .collect();
                    gd.adam_update(self, gradients, epoch)
                }
                println!(
                    "Epoch: {}, Theta Vector: {:#?}, Bias: {}, Cost: {}",
                    epoch,
                    gd.theta_vector,
                    gd.b,
                    gd.cost(gd.theta_vector.clone(), gd.b)
                );
                self.gd.update_neural_net()
            }
        }
    }
}