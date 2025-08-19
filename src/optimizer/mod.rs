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

                // If grad has an extra batch dimension, reduce it
                let reduced_grad = if grad.shape()[0] == param_mut.data.shape()[0] {
                    grad
                } else if param_mut.data.shape()[0] == 1 {
                    // Average along the batch dimension
                    grad.mean_axis(ndarray::Axis(0)).unwrap().insert_axis(ndarray::Axis(0))
                } else {
                    panic!(
                        "Gradient shape {:?} does not match parameter shape {:?}",
                        grad.shape(),
                        param_mut.data.shape()
                    );
                };
                param_mut.data -= &(reduced_grad * self.lr);
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
