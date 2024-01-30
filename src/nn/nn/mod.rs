pub mod nn {

    use crate::nn::layer::layer::Layer;
    use ndarray::{Array1, Array2, ArrayBase};
    use ndarray::Axis;

    pub struct NeuralNetwork {
        pub layers: Vec<Layer>,
        pub num_layers: usize,
        pub alpha: f64,
        pub weights: Vec<Array1<f64>>,
        pub biases: Vec<Array1<f64>>,
    }

    impl NeuralNetwork {
        pub fn new(layers: Vec<Layer>, alpha: f64) -> Self {
            let num_layers = layers.len();
            let weights: Vec<Array1<f64>> = layers.iter().map(|layer| layer.get_weights()).collect();
            let biases: Vec<Array1<f64>> = layers.iter().map(|layer| layer.get_biases()).collect();
            Self {
                layers,
                num_layers,
                alpha,
                weights,
                biases,
            }
        }

        pub fn add_layer(&mut self, layer: Layer, location: usize) {
            self.layers.insert(location, layer);
            self.num_layers += 1;
        }

        pub fn predict(&self, input_layer: &Array1<f64>) -> Array1<f64> {
            let mut outputs: Vec<Array1<f64>> = Vec::new();
            let mut input: Array1<f64> = input_layer.clone();
            for layer in &self.layers {
                let output = layer.dense_numpy(&input);
                outputs.push(output.clone());
                input = output;
            }
            println!("{:?}", outputs);
            outputs.last().unwrap().clone()
        }

        pub fn mse(&self, y_hats: &Array1<f64>, y_actuals: &Array1<f64>) -> f64 {
            let diff = y_hats - y_actuals;
            let sum_squared = diff.mapv(|x| x.powi(2)).sum();
            sum_squared / (2.0 * y_actuals.len() as f64)
        }

        pub fn sigmoid_cost(&self, y_hats: &Array1<f64>, y_actuals: &Array1<f64>) -> f64 {
            let sum = y_actuals * y_hats.mapv(|x| x.log2()) + (1.0 - y_actuals) * (1.0 - y_hats).mapv(|x| x.log2());
            -sum.sum() / y_actuals.len() as f64
        }

        pub fn multiclass_cost(&self, y_hats: &Array2<f64>, y_actuals: &Array2<f64>) -> f64 {
            let sum = y_actuals.dot(&y_hats.t()).sum();
            -sum / y_actuals.len() as f64
        }

        pub fn forward_prop(&self, x: &Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
            let m = x.shape()[1];
            let mut a: Vec<Array1<f64>> = Vec::with_capacity(self.num_layers + 1);
            let mut z: Vec<Array1<f64>> = Vec::with_capacity(self.num_layers + 1);
            a.push(x.clone());
            z.push(Array1::zeros(0)); // Dummy value
            for (i, layer) in self.layers.iter().enumerate() {
                let input_shape: &[usize] = a[i].shape();
                let expected_shape: (usize, usize) = if i == 0 {
                    (layer.num_weights, m)
                } else {
                    (self.layers[i - 1].num_nodes, m)
                };
                assert_eq!(&input_shape[..], &[expected_shape.0, expected_shape.1], "Input shape mismatch");
                let z_push = layer.sigmoid_z(&a[i]);
                z.push(z_push);
                a.push(layer.sigmoid_z(&z[i + 1]));
                println!("Forward Pass layer {}", i + 1);
            }
            (z, a)
        }
    
        pub fn back_prop(&self, activations: &Vec<Array2<f64>>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array1<f64>>) {
            let m = y.shape()[0];
            let mut dz: Vec<Array2<f64>> = Vec::with_capacity(self.num_layers);
            let mut dw: Vec<Array2<f64>> = Vec::with_capacity(self.num_layers);
            let mut db: Vec<Array1<f64>> = Vec::with_capacity(self.num_layers);
            for l in (1..=self.num_layers).rev() {
                let dz_curr = if l == self.num_layers {
                    activations[l].clone() - y
                } else {
                    let dz_clone = dz.clone();
                    let da: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>> = self.layers[l - 1].get_weights().t().dot(&dz_clone[self.num_layers - l].clone().reversed_axes().view());
                    let g_prime: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>> = activations[l].mapv(|x| x * (1.0 - x));
                    da * g_prime
                };
                let dz_curr_clone = dz_curr.clone();
                dz.push(dz_curr);
                dw.push((dz_curr_clone.dot(&activations[l - 1].t())) / m as f64);
                db.push(dz_curr_clone.sum_axis(Axis(1)) / m as f64);
                println!("Backward Pass layer {}", self.num_layers - l + 1);
            }
            dz.reverse();
            dw.reverse();
            db.reverse();
            (dw, db)
        }
    
        pub fn gradient_descent(&mut self, cost_fun: &str, x: Array1<f64>, y: Array1<f64>, epochs: usize) {
            let mut costs: Vec<f64> = Vec::new();
            let y_clone: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>> = y.clone();
            let y_clone_2: ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>> = y_clone.clone();
            match cost_fun {
                "MSE" => {
                    // need to implement logic here
                }
                "sigmoid" => {
                    for e in 0..epochs {
                        // Forward Prop
                        let a: Vec<ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>> = self.forward_prop(&x).1;
                        // Calculate cost
                        //let y_clone = y.clone();
                        let cost = self.sigmoid_cost(&a[a.len() - 1], &y_clone);
                        println!("Cost on epoch {}: {}", e, cost);
                        costs.push(cost);
                        // Backward Prop
                        let (d_w, d_b) = self.back_prop(
                            &a.into_iter().map(|arr| arr.insert_axis(Axis(1))).collect(),
                            &y_clone_2.clone().insert_axis(Axis(0)),
                        ); 
                        for (i, layer) in self.layers.iter_mut().enumerate() {
                            let w_new = layer.get_weights() - self.alpha * &d_w[i];
                            println!("New Weights on epoch {} in layer {}: {:?}", e, i + 1, w_new);
                            let b_new = layer.get_biases() - self.alpha * &d_b[i];
                            println!("New Biases on epoch {} in layer {}: {:?}", e, i + 1, b_new);
                            layer.set_all_weights(
                                w_new
                                    .into_iter()
                                    .map(|arr| ArrayBase::<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>::from_vec(vec![arr]).into_owned())
                                    .collect(),
                                b_new.into_raw_vec(),
                            );
                            
                        }
                        println!("Costs: {:?}", costs);
                    }
                }
                "multiclass" => {
                    // need to implement logic here
                }
                _ => {
                    println!("Invalid cost function");
                }
            }
        }
    }
}
