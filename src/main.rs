mod nn {
    pub mod layer;
    pub mod nn;
    pub mod node;
}

mod gradient_descent {
    pub mod obj;
    pub mod stochastic;
    pub mod batch;
}

mod cnn {
    pub mod image_tensor;
}

mod adam;

use nn::layer::layer::Layer;
use nn::nn::nn::NeuralNetwork;
use ndarray::{Array1, arr1};
use rand::Rng;

fn gen_data(size: usize, correlation: f64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();
    let x: Array1<f64> = Array1::linspace(0.0, 10.0, size);
    let y: Array1<f64> = x.mapv(|xi| correlation * xi + rng.gen_range(-1.0..1.0));
    (x, y)
}

fn main() {
    let mut layer_1: Layer = Layer::new(2);
    let mut layer_2: Layer = Layer::new(1);

    let w_1: Vec<Array1<f64>> = vec![
        arr1(&[0.5, 0.6]),
        arr1(&[0.7, 0.8]),
    ];
    let b_1: Vec<f64> = vec![0.07, 0.05];
    layer_1.set_all_weights(w_1, b_1);

    let w_2: Vec<Array1<f64>> = vec![
        arr1(&[0.05, 0.2])
    ];
    let b_2: Vec<f64> = vec![0.03];
    layer_2.set_all_weights(w_2, b_2);

    let mut nn: NeuralNetwork = NeuralNetwork::new(vec![layer_1, layer_2], 0.001);
    let train_data: (ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>, ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>) = gen_data(100, 0.8);
    nn.gradient_descent("sigmoid", train_data.0, train_data.1, 100)
}