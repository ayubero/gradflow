use std::{cell::RefCell, rc::Rc};

use crate::tensor::Tensor;

pub struct SGD {
    pub params: Vec<Rc<RefCell<Tensor>>>,
    pub lr: f64,
}

impl SGD {
    pub fn step(&self) {
        for param in &self.params {
            if let Some(ref mut grad) = param.borrow_mut().grad {
                param.borrow_mut().data -= &(grad.clone() * self.lr);
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
