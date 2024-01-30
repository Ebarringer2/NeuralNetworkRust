pub mod layer {
    extern crate ndarray;
    use ndarray::Array1;
    use crate::nn::node::node::Node;
    use rand::Rng;

    #[derive(Debug)]
    pub struct Layer {
        pub nodes: Vec<Node>,
        pub num_nodes: usize,
        pub num_weights: usize,
    }

    impl Layer {
        pub fn new(num_nodes: usize) -> Self {
            let weights:ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 1]>>  = Array1::from_vec(vec![rand::thread_rng().gen_range(0.0..1.0)]);
            let b: f64 = rand::thread_rng().gen_range(0.0..1.0);
            let nodes: Vec<Node> = (0..num_nodes).map(|_| Node::new(weights.clone(), b)).collect();
            let num_weights: usize = if num_nodes > 0 { nodes[0].weights.len() } else { 0 };

            Self {
                nodes,
                num_nodes,
                num_weights,
            }
        }

        pub fn set_weights(&mut self, index: usize, weights: Array1<f64>, bias: f64) {
            self.nodes[index].set_weights(weights, bias);
        }

        pub fn set_all_weights(&mut self, weights: Vec<Array1<f64>>, biases: Vec<f64>) {
            assert_eq!(weights.len(), biases.len(), "Weights and biases lists must be the same length");

            self.num_weights = weights[0].len();

            for i in 0..self.num_nodes {
                self.nodes[i].set_weights(weights[i].clone(), biases[i]);
            }
        }

        pub fn execute_layer(&self, input_layer: &Array1<f64>) -> Array1<f64> {
            let num_features = input_layer.len();

            assert_eq!(num_features, self.num_weights, "Number of weights does not match number of features");

            self.nodes
                .iter()
                .map(|node| node.sigmoid_actualize(input_layer))
                .collect()
        }

        pub fn dense_numpy(&self, input_layer: &Array1<f64>) -> Array1<f64> {
            let z = self.get_weights().dot(input_layer) + self.get_biases();
            self.sigmoid(&z)
        }

        pub fn sigmoid_z(&self, input_layer: &Array1<f64>) -> Array1<f64> {
            println!("Input layer shape:  {:?}", input_layer.shape());
            println!("Node weights shape: {:?}", self.get_weights().shape());
            self.get_weights().dot(input_layer) + self.get_biases()
        }

        pub fn sigmoid(&self, input: &Array1<f64>) -> Array1<f64> {
            input.mapv(|x| 1.0 / (1.0 + f64::exp(-x)))
        }

        pub fn print_layer(&self) {
            for (i, node) in self.nodes.iter().enumerate() {
                println!("Node {}: {:?}", i + 1, node);
            }
        }

        pub fn get_weights(&self) -> Array1<f64> {
            let weights: Vec<f64> = self.nodes.iter().flat_map(|node| node.get_weights().to_vec()).collect();
            Array1::from(weights)
        }
        
         
        pub fn get_biases(&self) -> Array1<f64> {
            self.nodes.iter().map(Node::get_bias).collect()
        }

        pub fn get_nodes(&self) -> &Vec<Node> {
            &self.nodes
        }
    }
}
