pub mod nn {
    extern crate ndarray;
    //use core::num;
    use ndarray::{Array, Array2, Axis};
    use crate::nn::layer::layer::Layer;
    use std::thread;
    use std::time::Duration;

    #[derive(Clone)]
    pub struct NeuralNetwork {
        pub layers: Vec<Layer>,
        pub num_layers: usize,
        pub alpha: f64,
        pub weights: Vec<Vec<Array2<f64>>>,
        pub biases: Vec<Vec<Array2<f64>>>
    }

    impl NeuralNetwork {

        /// Creates a new Neural Network object with an input Vec<Layer> and a learning rate
        pub fn new(layers: Vec<Layer>, alpha: f64) -> NeuralNetwork {
            let num_layers: usize = layers.len();
            let weights = layers.iter().map(|layer| layer.get_weights().clone()).collect();
            let biases = layers.iter().map(|layer| layer.get_biases().clone()).collect();
            println!("CREATED NEURAL NETWORK OBJECT WITH NUM_LAYERS: {}", num_layers);
            NeuralNetwork {
                layers,
                num_layers,
                alpha,
                weights,
                biases
            }
        }
        pub fn add_layer(&mut self, layer: Layer, location: usize) {
            self.layers.insert(location, layer);
            self.num_layers += 1;
        }
        pub fn predict(&self, input_layer: Vec<f64>) -> f64 {
            println!("\n\nCALCULATING PREDICTION");
            thread::sleep(Duration::from_secs(2));
            let mut outputs: Vec<f64> = Vec::new();
            let input: Vec<f64> = input_layer;
            let mut index: i64 = 1;
            for layer in &self.layers {
                println!("activating nodes in hidden layer {}", index);
                index += 1;
                if layer.activation_function == "reLU" {
                    let output: Vec<f64> = layer.get_nodes().iter().map(|node| node.reLU_activate(input.clone())).collect();
                    outputs.extend(output.clone())
                } else if layer.activation_function == "sigmoid" {
                    let output: Vec<f64> = layer.get_nodes().iter().map(|node| node.sigmoid_actualize(input.clone())).collect();
                    outputs.extend(output.clone());
                }
            }
            println!("{:?}", outputs);
            return outputs[outputs.len() - 1];
        }
        pub fn print_layers(&self) {
            for i in 0..self.num_layers {
                println!("Layer {}", i + 1);
                self.layers[i].print_layer();

            }
        }
        pub fn mut_clone(&mut self) -> NeuralNetwork {
            NeuralNetwork {
                layers: self.layers.iter().map(|layer| layer.clone()).collect(),
                num_layers: self.num_layers,
                alpha: self.alpha,
                weights: self.weights.clone(),
                biases: self.biases.clone()
            }
        }

        pub fn forward_prop(&self, x: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
            let m = x.shape()[1];
            let mut a: Vec<Array2<f64>> = vec![x.clone()];
            let mut z: Vec<Array2<f64>> = vec![Array::zeros((0, 0))];
    
            // Forward Prop
            for (i, layer) in self.layers.iter().enumerate() {
                if i == 0 {
                    assert_eq!(a[i].shape(), (self.layers[i].num_weights, m));
                } else {
                    println!("Inputs shape: {:?}", a[i].shape());
                    println!(
                        "Should be: {:?}",
                        (self.layers[i - 1].get_weights().shape()[1], m)
                    );
                    assert_eq!(
                        a[i].shape(),
                        (self.layers[i - 1].get_weights().shape()[1], m)
                    );
                }
                z.push(layer.sigmoid_z(&a[i]));
                a.push(layer.sigmoid_a(&z[i + 1]));
                println!("Forward Pass layer {}", i + 1);
            }
    
            (z, a)
        }
    
        pub fn back_prop(
            &self,
            activations: Vec<Array2<f64>>,
            y: &Array2<f64>,
        ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
            let m = y.shape()[1];
            let mut dz: Vec<Array2<f64>> = Vec::new();
            let mut dw: Vec<Array2<f64>> = Vec::new();
            let mut db: Vec<Array2<f64>> = Vec::new();
    
            for l in (1..=self.num_layers).rev() {
                let dz_curr: Array2<f64>;
                if l == self.num_layers {
                    dz_curr = activations[l].clone() - y;
                    dz.push(dz_curr);
                } else {
                    let da = self.layers[l].get_weights().t().dot(&dz[l - 1]);
                    let g_prime = activations[l].mapv(|x| x * (1.0 - x));
                    dz_curr = da * &g_prime;
                    dz.push(dz_curr);
                }
    
                dw.push((1.0 / m as f64) * dz_curr.dot(&activations[l - 1].t()));
                db.push((1.0 / m as f64) * dz_curr.sum_axis(Axis(1)).insert_axis(Axis(1)));
    
                println!("Backward Pass layer {}", l);
            }
    
            (dw, db)
        }
    
        pub fn gradient_descent(&mut self, cost_fun: &str, x: &Array2<f64>, y: &Array2<f64>, epochs: usize) {
            let mut costs: Vec<f64> = Vec::new();
    
            match cost_fun {
                "MSE" => {

                }
                "sigmoid" => {
                    for e in 0..epochs {
                        let a = self.forward_prop(x).1;
                        let cost = self.sigmoid_cost(&a[a.len() - 1], y);
                        println!("Cost on epoch {}: {}", e, cost);
                        costs.push(cost);
                        let (dw, db) = self.back_prop(a, y);
                        for (i, layer) in self.layers.iter_mut().enumerate() {
                            let w_new = layer.get_weights() - self.alpha * &dw[i];
                            println!("New Weights on epoch {} in layer {}: {:?}", e, i + 1, w_new);
                            let b_new = layer.get_weights() - self.alpha * &db[i];
                            println!("New Biases on epoch {} in layer {}: {:?}", e, i + 1, b_new);
                            layer.set_all_weights(w_new, b_new, false);
                        }
                        println!("Costs: {:?}", costs);
                    }
                }
                "multiclass" => {
                }
                _ => {
                }
            }
        }
        
    }
}