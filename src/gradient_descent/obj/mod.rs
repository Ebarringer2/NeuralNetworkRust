// contains the GD struct

// This code needs to be updated, and will likely just be retired because the neural network already 
// has back propagation and gradient descent as its own methods. I will comment this code out in order
// to avoid compiler errors, but it can be uncommented for whatever reason necessary

/* 
pub mod obj {
    use crate::adam::adam::Adam;
    use crate::nn::nn::nn::NeuralNetwork;
    use rand::Rng;

    #[derive(Clone)]
    pub struct GradientDescent {
        pub neural_net: NeuralNetwork,
        pub theta_matrix: Vec<Vec<f64>>,
        pub b: f64,
        pub learning_rate: f64,
        pub num_predictors: usize,
        pub x_train: Vec<f64>,
        pub y_train: Vec<f64>
    }

    impl GradientDescent {

        /// Creates a new GradientDescent object
        pub fn new(neural_net: NeuralNetwork, x_train: Vec<f64>, y_train: Vec<f64>, num_predictors: usize, learning_rate: f64) -> GradientDescent {

            let mut theta_matrix: Vec<Vec<f64>> = Vec::new();
            let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
            for layer in neural_net.layers.iter() {
                let num_nodes = layer.num_nodes;
                let mut weights: Vec<f64> = Vec::with_capacity(num_nodes + 1);
                weights.extend((0..=num_nodes).map(|_| rng.gen_range(0.0..1.0)));
                theta_matrix.push(weights);
            }
            
            let b: f64 = rng.gen_range(0.0..1.0);

            GradientDescent {
                neural_net,
                theta_matrix,
                b,
                learning_rate,
                num_predictors,
                x_train,
                y_train
            }
        }
        
        pub fn update_neural_net(&mut self) {
            //println!("Updating Neural Network");
            let self_clone = self.clone();
            for (i, layer) in self.neural_net.layers.iter_mut().enumerate() {
                //self_clone.neural_net.print_layers();
                let (weights, bias) = self_clone.get_params_for_layer(i);
                //println!("Weights: {:#?}", weights);
                //println!("Bias: {:?}", bias);
                layer.set_all_weights(weights.clone(), bias);
                //println!("Set layer weights: {:#?}", weights);
                //println!("Set layer bias: {:?}", bias);
            }
        }

        pub fn get_params_for_layer(&self, layer_index: usize) -> (Vec<f64>, f64) {
            let weights: Vec<f64> = self.theta_matrix[layer_index].clone();
            let _bias_index: usize = layer_index;
            let bias: f64 = self.theta_matrix[layer_index].last().cloned().unwrap_or(0.0);
            (weights, bias)
        }

        pub fn h(&self, X: Vec<f64>) -> f64 {
            let mut result: f64 = 0.0;
            for (i, x) in X.iter().enumerate() {
                result += self.theta_matrix[i].iter().map(|&theta| theta * x).sum::<f64>();
            }
            result + self.b
        }

        pub fn h_vectorized(&self, X: Vec<f64>) -> f64 {
            self.theta_matrix
                .iter()
                .flat_map(|theta_row| theta_row.iter())
                .zip(X.iter())
                .fold(0.0, |acc, (&theta, &x)| acc + theta * x) + self.b
        }
        
        
        pub fn h_given_params(&self, X: Vec<f64>, theta_matrix: Vec<Vec<f64>>, b: f64) -> f64 {
            theta_matrix
                .iter()
                .flat_map(|layer| layer.iter().zip(X.iter()))
                .fold(0.0, |acc, (&theta, &x)| acc + theta * x)
                + b
        }
        
        pub fn get_y(&self) -> f64 {
            self.b.clone()
        }

        pub fn get_params(&self) -> Vec<Vec<f64>> {
            self.theta_matrix.clone()
        }

        pub fn train_data(&self) -> Vec<(Vec<f64>, f64)> {
            let output: Vec<(Vec<f64>, f64)> = self.x_train
                .iter()
                .cloned()
                .zip(self.y_train.iter().cloned())
                .map(|(predictors, output)| (vec![predictors], output))
                .collect();
            //println!("Output of training data generation: {:?}", output.clone());
            output
        }

        /// Returns the costs for each layer in the Neural Network as a vector of floats
        pub fn cost(&self, theta_matrix: Vec<Vec<f64>>, b: f64) -> Vec<f64> {
            let m: f64 = self.x_train.len() as f64;
            let mut costs: Vec<f64> = Vec::new();
            for (predictors, output) in self.train_data() {
                let prediction = self.h_given_params(predictors.to_vec(), theta_matrix.clone(), b);
                let individual_cost = (prediction - output).powf(2.0) / (2.0 * m);
                costs.push(individual_cost);
            }
            costs
        }
        
        /// MSE loss
        pub fn mse_loss(y_pred: f64, y_true: f64) -> f64 {
            let error = y_pred - y_true;
            error * error
        }
        
        pub fn adam_update(&mut self, adam: &mut Adam, gradients: &Vec<f64>, epoch: usize) {
            //println!("ADAM UPDATING");
            //println!("USING GRADIENTS: {:?}", gradients.clone());
            let mut m: Vec<f64> = vec![0.0; self.num_predictors];
            let mut v: Vec<f64> = vec![0.0; self.num_predictors];
            let beta_1_pow: f64 = adam.beta_1.powi(epoch as i32);
            let beta_2_pow: f64 = adam.beta_2.powi(epoch as i32);
            for i in 0..self.num_predictors {
                if i < gradients.len() || i == gradients.len() {
                    //println!("Index: {}", i.clone());
                    //println!("Gradients Len: {}", gradients.len());
                    //println!("Theta Matrix Len: {}", self.clone().theta_matrix.len());
                    m[i] = adam.beta_1 * m[i] + (1.0 - adam.beta_1) * gradients[i];
                    v[i] = adam.beta_2 * v[i] + (1.0 - adam.beta_2) * gradients[i].powi(2);
                    let m_hat: f64 = m[i] / (1.0 - beta_1_pow);
                    let v_hat: f64 = v[i] / (1.0 - beta_2_pow);
                    if i < self.clone().theta_matrix.len() {
                        self.theta_matrix[i].iter_mut().for_each(|theta| {
                            *theta -= self.learning_rate * m_hat / (v_hat.sqrt() + adam.epsilon);
                        });
                    }                    
                } else {
                    break;
                }
            }
            let gradient_b: f64 = gradients.iter().sum();
            //println!("GRADIENTS Len: {}", gradients.len());
            adam.m_b.iter_mut().enumerate().for_each(|(i, m_b)| {
                *m_b = adam.beta_1 * *m_b + (1.0 - adam.beta_1) * gradients[i];
            });
            adam.v_b.iter_mut().zip(gradients.iter()).for_each(|(v, &grad)| {
                *v = adam.beta_2 * *v + (1.0 - adam.beta_2) * grad.powi(2);
            });
            let m_b_hat: f64 = adam.m_b.iter().sum::<f64>() / (1.0 - beta_1_pow);
            let v_b_hat: f64 = adam.v_b.iter().sum::<f64>() / (1.0 - beta_2_pow);
            self.theta_matrix.iter_mut().for_each(|theta_row| {
                theta_row.iter_mut().for_each(|theta| {
                    *theta -= self.learning_rate * m_b_hat / (v_b_hat.sqrt() + adam.epsilon);
                });
            });            
            self.b -= self.learning_rate * gradient_b / (v_b_hat.sqrt() + adam.epsilon);
        }
        
    }
}
*/