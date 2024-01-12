pub mod stochastic {
    
    use crate::gradient_descent::obj::obj::GradientDescent;

    pub fn stochastic_vect(mut gd: GradientDescent, epochs: i32) {
        let m: f64 = gd.x_train.len() as f64;

        for i in 0..epochs {
            for (predictor, output) in gd.train_data() {
                let error = gd.h(predictor.clone()) - output;
                for j in 0..gd.num_predictors {
                    gd.theta_vector[j] -= (gd.learning_rate / m) * error * predictor[j]
                }
                gd.b -= (gd.learning_rate / m) * error
            }
            println!("Epoch: {}, Theta Vector: {:#?}, Cost: {}", i, gd.theta_vector, gd.cost(gd.theta_vector.clone(), gd.b))
        }
    }

}