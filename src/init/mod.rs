use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::nn::Linear;

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
