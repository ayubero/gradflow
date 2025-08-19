use std::{cell::RefCell, rc::Rc};

use ndarray::Axis;

use crate::tensor::Tensor;

pub struct SGD {
    pub params: Vec<Rc<RefCell<Tensor>>>,
    pub lr: f64,
}

impl SGD {
    pub fn step(&self) {
        for param in &self.params {
            let mut param_mut = param.borrow_mut();
            if param_mut.grad.is_some() {
                let grad = param_mut.grad.as_ref().unwrap().clone();
                param_mut.data -= &(grad * self.lr);
            }
        }
    }

    pub fn zero_grad(&self) {
        for param in self.params.iter() {
            if let Some(ref mut grad) = param.borrow_mut().grad {
                *grad = ndarray::ArrayD::zeros(grad.raw_dim());
            }
        }
    }
}
