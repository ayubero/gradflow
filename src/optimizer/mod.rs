use std::{cell::RefCell, rc::Rc};

use crate::tensor::Tensor;

pub struct SGD {
    pub lr: f64,
}

impl SGD {
    pub fn step(&self, params: Vec<Rc<RefCell<Tensor>>>) {
        for param in params {
            if let Some(ref mut grad) = param.borrow_mut().grad {
                param.borrow_mut().data -= &(grad.clone() * self.lr);
            }
        }
    }
}
