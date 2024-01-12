// contains the GD struct

pub mod obj {
    use crate::adam::adam::Adam;
    use crate::nn::nn::nn::NeuralNetwork;
    use rand::Rng;

    #[derive(Clone)]
    pub struct GradientDescent {
        pub neural_net: NeuralNetwork,
        pub theta_vector: Vec<f64>,
        pub b: f64,
        pub learning_rate: f64,
        pub num_predictors: usize,
        pub x_train: Vec<f64>,
        pub y_train: Vec<f64>
    }

    impl GradientDescent {

        /// Creates a new GradientDescent object
        pub fn new(neural_net: NeuralNetwork, x_train: Vec<f64>, y_train: Vec<f64>, num_predictors: usize, learning_rate: f64) -> GradientDescent {

            let mut theta_vector: Vec<f64> = Vec::with_capacity(num_predictors);
            let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
            theta_vector.extend((0..num_predictors).map(|_| rng.gen_range(0.0..1.0)));
            
            let b: f64 = rng.gen_range(0.0..1.0);

            GradientDescent {
                neural_net,
                theta_vector,
                b,
                learning_rate,
                num_predictors,
                x_train,
                y_train
            }
        }
        
        pub fn update_neural_net(&mut self) {
            println!("Updating Neural Network");
            let self_clone = self.clone();
            for (i, layer) in self.neural_net.layers.iter_mut().enumerate() {
                self_clone.neural_net.print_layers();
                let (weights, bias) = self_clone.get_params_for_layer(i);
                println!("Weights: {:#?}", weights);
                println!("Bias: {:?}", bias);
                layer.set_all_weights(weights.clone().into_iter().flatten().collect(), bias.clone(), true);
                println!("Set layer weights: {:#?}", weights);
                println!("Set layer bias: {:?}", bias);
            }
        }

        pub fn get_params_for_layer(&self, layer_index: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
            let weights_start: usize = layer_index * self.num_predictors;
            let weights_end: usize = std::cmp::min(weights_start + self.num_predictors, self.theta_vector.len());
            let bias_index: usize = self.theta_vector.len() - 1;
            println!("Weights Start: {}", weights_start);
            println!("Weights End: {}", weights_end);
            println!("Bias Index: {}", bias_index);
            let weights: Vec<Vec<f64>> = self.theta_vector[weights_start..weights_end]
                .chunks(self.num_predictors)
                .map(|chunk| chunk.to_vec())
                .collect();
            let bias: Vec<f64> = vec![self.theta_vector[bias_index]];
            (weights, bias)
        }

        pub fn h(&self, X: Vec<f64>) -> f64 {
            let mut result: f64 = 0.0;
            for (i, x) in X.iter().enumerate() {
                result += (self.theta_vector[i] * x).clone()
            }
            result + self.b
        }

        pub fn h_vectorized(&self, X: Vec<f64>) -> f64 {
            self.theta_vector.iter().zip(X.iter()).fold(0.0, |acc, (&theta, &x)| acc + theta * x) + self.b
        }
        
        pub fn h_given_params(&self, X: Vec<f64>, theta_vector: Vec<f64>, b: f64) -> f64 {
            theta_vector.iter().zip(X.iter()).fold(0.0, |acc, (&theta, &x)| acc + theta * x) + b
        }

        pub fn get_y(&self) -> f64 {
            self.b.clone()
        }

        pub fn get_params(&self) -> Vec<f64> {
            self.theta_vector.clone()
        }

        pub fn train_data(&self) -> Vec<(Vec<f64>, f64)> {
            self.x_train
                .iter()
                .cloned()
                .zip(self.y_train.iter().cloned())
                .map(|(predictors, output)| (vec![predictors], output))
                .collect()
        }

        pub fn cost(&self, theta_vector: Vec<f64>, b: f64) -> f64 {
            let m: f64 = self.x_train.len() as f64;
            let mut cost: f64 = 0.0;
            for (predictors, output) in self.train_data() {
                cost += (self.h_given_params(predictors.to_vec(), theta_vector.clone(), b) - output).powf(2.0);
            }
            cost * 1.0 / (2.0 * m)
        }   

        pub fn adam_update(&mut self, adam: &mut Adam, gradients: Vec<f64>, epoch: usize) {
            let mut m: Vec<f64> = vec![0.0; self.num_predictors];
            let mut v: Vec<f64> = vec![0.0; self.num_predictors];
            let beta_1_pow: f64 = adam.beta_1.powi(epoch as i32);
            let beta_2_pow: f64 = adam.beta_2.powi(epoch as i32);
            for i in 0..self.num_predictors {
                if i < gradients.len() {
                    m[i] = adam.beta_1 * m[i] + (1.0 - adam.beta_1) * gradients[i];
                    println!("EPOCH {} Updating m[i]", i);
                    v[i] = adam.beta_2 * v[i] + (1.0 - adam.beta_2) * gradients[i].powi(2);
                    println!("EPOCH {} Updated v[i]", i);
                    let m_hat: f64 = m[i] / (1.0 - beta_1_pow);
                    println!("EPOCH {} Calculated m_hat", i);
                    let v_hat: f64 = v[i] / (1.0 - beta_2_pow);
                    println!("EPOCH {} Calculated v_hat", i);
                    self.theta_vector[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + adam.epsilon);
                    println!("EPOCH {} Updated theta vector", i);
                } else {
                    println!("Made it to break case");
                    break;
                }
            }
            let gradient_b: f64 = gradients.iter().map(|&x| x).sum();
            adam.m_b = adam.beta_1 * adam.m_b + (1.0 - adam.beta_1) * gradient_b;
            adam.v_b = adam.beta_2 * adam.v_b + (1.0 - adam.beta_2) * gradient_b.powi(2);
            let m_b_hat: f64 = adam.m_b / (1.0 - beta_1_pow);
            let v_b_hat: f64 = adam.v_b / (1.0 - beta_2_pow);
            self.b -= self.learning_rate * m_b_hat / (v_b_hat.sqrt() + adam.epsilon);
        }
    }
}