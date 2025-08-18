use std::{cell::RefCell, rc::Rc};

use ndarray::ArrayD;

use crate::tensor::{add, matmul, Tensor};

pub trait Module {
    fn forward(&self, input: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>>;
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>>;
}

// ====================
// Sequential Container
// ====================

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, mut input: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

// ============
// Linear Layer
// ============

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
}

impl Module for Linear {
    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        let wx = matmul(&self.weights, &x);
        add(&wx, &self.bias)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![Rc::clone(&self.weights), Rc::clone(&self.bias)]
    }
}

