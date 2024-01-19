pub mod layer {
    use crate::nn::node::node::Node;
    use rand::Rng;

    #[derive(Clone)]
    pub struct Layer {
        pub nodes: Vec<Node>,
        pub num_nodes: usize,
        pub activation_function: String
    }

    impl Layer {
        
        /// Creates a new Layer object with a given number of nodes
        pub fn new(num_nodes: usize, num_features: usize, activation_function: String) -> Layer {
            let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
            let bias: f64 = rng.gen::<f64>();
            let nodes: Vec<Node> = (0..num_nodes).map(|_| {
                let weights: Vec<f64> = (0..num_features).map(|_| rng.gen::<f64>()).collect();
                Node::new(weights, bias)
            }).collect();
            Layer {
                nodes,
                num_nodes,
                activation_function
            }
        }
        pub fn set_weights(&mut self, index: usize, weights: Vec<f64>, bias: f64) {
            self.nodes[index].set_weights(weights, bias)
        }

        /// set_zeros paramater is for forceful error adapation, sets the weights to a vector
        /// of 0s with a length of the layer size
        pub fn set_all_weights(&mut self, mut weights: Vec<f64>, bias: f64, set_zeros: bool) {
            if self.num_nodes != weights.len() {
                if set_zeros {
                    //println!("OVERRIDING: SETTING WEIGHTS TO VEC<0s>");
                    weights = vec![0.0; self.num_nodes]
                } else {
                    //println!("WEIGHTS len: {}", weights.len());
                    //println!("BIAS: {}", bias);
                    //println!("LAYER size: {}", self.num_nodes);
                    panic!("Weights and bias lists must match layer size!");
                }
            }
            for i in 0..self.num_nodes {
                self.nodes[i].set_weights(weights.clone(), bias);
                //println!("Done setting weights for Node {}", i);
            }
        }
        
        pub fn execute(&self, input_layer: Vec<f64>) -> Vec<f64> {
            let num_features: usize = input_layer.len();
            let mut a_vector: Vec<f64> = Vec::new();
            for node in &self.nodes {
                if num_features != node.weights.len() {
                    panic!("Number of weights does not match number of features");
                }
                if self.activation_function == "sigmoid" {
                    a_vector.push(node.sigmoid_actualize(input_layer.clone()));
                } else if self.activation_function == "reLU" {
                    a_vector.push(node.reLU_activate(input_layer.clone()));
                }
            }
            a_vector
        }
        pub fn print_layer(&self) {
            for i in 0..self.num_nodes {
                println!("Node {}: {}", i + 1, self.nodes[i])
            }
        }
        pub fn get_nodes(&self) -> Vec<Node> {
            self.nodes.clone()
        }

        pub fn back_propagate(&self) {
            
            
        }
    }
}
