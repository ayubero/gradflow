use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::nn::{Conv2d, Linear};

pub enum InitType {
    XavierUniform,
    XavierNormal,
    KaimingUniform,
    KaimingNormal,
}

pub trait Initializable {
    fn reset_parameters(&mut self, init: &InitType);
}

impl Initializable for Linear {
    fn reset_parameters(&mut self, init: &InitType) {
        let fan_in = self.weights.borrow().data.shape()[1];
        let fan_out = self.weights.borrow().data.shape()[0];
        let mut rng = rand::thread_rng();

        match init {
            InitType::XavierUniform => {
                let bound = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
                for w in self.weights.borrow_mut().data.iter_mut() {
                    *w = rng.gen_range(-bound..bound);
                }
            }
            InitType::XavierNormal => {
                let std = (2.0 / (fan_in as f64 + fan_out as f64)).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                for w in self.weights.borrow_mut().data.iter_mut() {
                    *w = normal.sample(&mut rng);
                }
            }
            InitType::KaimingUniform => {
                let bound = (6.0 / fan_in as f64).sqrt();
                for w in self.weights.borrow_mut().data.iter_mut() {
                    *w = rng.gen_range(-bound..bound);
                }
            }
            InitType::KaimingNormal => {
                let std = (2.0 / fan_in as f64).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                for w in self.weights.borrow_mut().data.iter_mut() {
                    *w = normal.sample(&mut rng);
                }
            }
        }
    }
}

impl Initializable for Conv2d {
    fn reset_parameters(&mut self, init: &InitType) {
        let shape = self.weights.borrow().data.shape().to_vec();
        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_h = shape[2];
        let kernel_w = shape[3];
        
        let fan_in = in_channels * kernel_h * kernel_w;
        let fan_out = out_channels * kernel_h * kernel_w;
        
        let mut rng = rand::thread_rng();
        
        // Get mutable reference once and work with it
        let mut weights_ref = self.weights.borrow_mut();
        
        match init {
            InitType::XavierUniform => {
                let bound = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
                for w in weights_ref.data.iter_mut() {
                    *w = rng.gen_range(-bound..bound);
                }
            }
            InitType::XavierNormal => {
                let std = (2.0 / (fan_in as f64 + fan_out as f64)).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                for w in weights_ref.data.iter_mut() {
                    *w = normal.sample(&mut rng);
                }
            }
            InitType::KaimingUniform => {
                let bound = (6.0 / fan_in as f64).sqrt();
                for w in weights_ref.data.iter_mut() {
                    *w = rng.gen_range(-bound..bound);
                }
            }
            InitType::KaimingNormal => {
                let std = (2.0 / fan_in as f64).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                for w in weights_ref.data.iter_mut() {
                    *w = normal.sample(&mut rng);
                }
            }
        }
        
        // Release the weights borrow before borrowing bias
        drop(weights_ref);
        
        // Initialize bias to zero
        let mut bias_ref = self.bias.borrow_mut();
        for b in bias_ref.data.iter_mut() {
            *b = 0.0;
        }
    }
}