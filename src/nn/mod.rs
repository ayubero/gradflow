use std::{cell::RefCell, rc::Rc};

use ndarray::ArrayD;

use crate::tensor::{add, Tensor};

pub struct Linear {
    pub weights: Rc<RefCell<Tensor>>,
    pub bias: Rc<RefCell<Tensor>>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weights = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![out_features, in_features], 0.01), true)));
        let bias = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![out_features], 0.0), true)));

        Linear { weights, bias }
    }

    pub fn forward(&self, x: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        let wx = add(&self.weights, x);
        add(&wx, &self.bias)
    }
}
