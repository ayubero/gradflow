use std::{cell::RefCell, rc::Rc};

use ndarray::{ArrayD, Axis};

use crate::tensor::Tensor;

// ==========
// Schedulers
// ==========

pub trait LRScheduler {
    fn get_lr(&self, step: usize) -> f64;
}

pub struct LinearDecay {
    pub initial_lr: f64,
    pub total_steps: usize,
}

impl LRScheduler for LinearDecay {
    fn get_lr(&self, step: usize) -> f64 {
        let fraction = (step as f64 / self.total_steps as f64).min(1.0);
        self.initial_lr * (1.0 - fraction)
    }
}

pub struct ExponentialDecay {
    pub initial_lr: f64,
    pub gamma: f64,
}

impl LRScheduler for ExponentialDecay {
    fn get_lr(&self, step: usize) -> f64 {
        self.initial_lr * self.gamma.powi(step as i32)
    }
}

// ==========
// Optimizers
// ==========

pub struct SGD {
    pub params: Vec<Rc<RefCell<Tensor>>>,
    pub lr: f64, // Fallback learning rate
    pub scheduler: Option<Box<dyn LRScheduler>>,
    pub step_count: usize,
}

impl SGD {
    pub fn step(&mut self) {
        let lr = if let Some(scheduler) = &self.scheduler {
            scheduler.get_lr(self.step_count)
        } else {
            self.lr
        };

        for param in &self.params {
            let mut param_mut = param.borrow_mut();
            if let Some(grad) = &param_mut.grad {
                let reduced_grad = if grad.shape()[0] == param_mut.data.shape()[0] {
                    grad.clone()
                } else if param_mut.data.shape()[0] == 1 {
                    grad.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0))
                } else {
                    panic!(
                        "Gradient shape {:?} does not match parameter shape {:?}",
                        grad.shape(),
                        param_mut.data.shape()
                    );
                };
                param_mut.data -= &(reduced_grad * lr);
            }
        }

        self.step_count += 1;
    }

    pub fn zero_grad(&self) {
        for param in self.params.iter() {
            if let Some(ref mut grad) = param.borrow_mut().grad {
                *grad = ArrayD::zeros(grad.raw_dim());
            }
        }
    }
}


pub struct Adam {
    pub params: Vec<Rc<RefCell<Tensor>>>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub m: Vec<ArrayD<f64>>,
    pub v: Vec<ArrayD<f64>>,
    pub step_count: usize,
}

impl Adam {
    pub fn new(params: Vec<Rc<RefCell<Tensor>>>, lr: f64) -> Self {
        let m = params
            .iter()
            .map(|p| ArrayD::zeros(p.borrow().data.raw_dim()))
            .collect();
        let v = params
            .iter()
            .map(|p| ArrayD::zeros(p.borrow().data.raw_dim()))
            .collect();

        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m,
            v,
            step_count: 0,
        }
    }

    pub fn step(&mut self) {
        self.step_count += 1;
        let lr_t = self.lr;

        for (i, param) in self.params.iter().enumerate() {
            let mut param_mut = param.borrow_mut();
            if let Some(grad) = &param_mut.grad {
                // Reduce gradient tensor if needed
                let reduced_grad = if grad.shape()[0] == param_mut.data.shape()[0] {
                    grad.clone()
                } else if param_mut.data.shape()[0] == 1 {
                    grad.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0))
                } else {
                    panic!(
                        "Gradient shape {:?} does not match parameter shape {:?}",
                        grad.shape(),
                        param_mut.data.shape()
                    );
                };

                // Update biased first moment estimate
                self.m[i] = &self.m[i] * self.beta1 + reduced_grad.clone() * (1.0 - self.beta1);
                // Update biased second moment estimate
                self.v[i] = &self.v[i] * self.beta2 + reduced_grad.mapv(|x| x * x) * (1.0 - self.beta2);

                // Compute bias-corrected estimates
                let m_hat = &self.m[i] / (1.0 - self.beta1.powi(self.step_count as i32));
                let v_hat = &self.v[i] / (1.0 - self.beta2.powi(self.step_count as i32));

                // Update parameters
                param_mut.data -= &(&m_hat * lr_t / (v_hat.mapv(f64::sqrt) + self.epsilon));
            }
        }
    }

    pub fn zero_grad(&self) {
        for param in self.params.iter() {
            if let Some(ref mut grad) = param.borrow_mut().grad {
                *grad = ArrayD::zeros(grad.raw_dim());
            }
        }
    }
}