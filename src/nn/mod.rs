use std::{any::Any, cell::RefCell, rc::Rc};
use ndarray::{ArrayD, Axis};
use rayon::prelude::*;

use crate::{init::{InitType, Initializable}, tensor::{add, conv2d, flatten, matmul, maxpool2d, relu, sigmoid, tanh, Tensor}};

pub trait Module {
    fn forward(&self, input: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>>;
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ====================
// Sequential Container
// ====================

#[macro_export]
macro_rules! modules { // Macro to avoid writing Box for all the Modules in Sequential
    ($($module:expr),* $(,)?) => {
        vec![$(Box::new($module) as Box<dyn Module>),*]
    };
}

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }

    pub fn apply_init(&mut self, init: InitType) {
        for layer in self.layers.iter_mut() {
            if let Some(l) = layer.as_any_mut().downcast_mut::<Linear>() {
                l.reset_parameters(&init);
            } else if let Some(l) = layer.as_any_mut().downcast_mut::<Conv2d>() {
                l.reset_parameters(&init);
            }
        }
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ====================
// Polynomial expansion
// ====================

pub struct PolyExpand {
    pub degree: usize,
}

impl PolyExpand {
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }

    fn expand_row(&self, x: f64, y: f64) -> Vec<f64> {
        let mut expanded = vec![x, y];
        if self.degree >= 2 {
            expanded.push(x.powi(2));
            expanded.push(y.powi(2));
            expanded.push(x * y);
        }
        if self.degree >= 3 {
            expanded.push(x.powi(3));
            expanded.push(y.powi(3));
            expanded.push(x.powi(2) * y);
            expanded.push(x * y.powi(2));
        }
        expanded
    }
}

impl Module for PolyExpand {
    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        let input = &x.borrow().data;
        let mut rows = Vec::new();
        for row in input.rows() {
            let x = row[0];
            let y = row[1];
            rows.push(self.expand_row(x, y));
        }

        let arrays: Vec<ndarray::Array1<f64>> = rows
            .into_iter()
            .map(|r| ndarray::Array1::from(r))
            .collect();

        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

        let result = ndarray::stack(Axis(0), &views).unwrap();
        
        Rc::new(RefCell::new(Tensor::new(result.into_dyn(), false)))
    }
    
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
        let bias = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![1, out_features], 0.0), true)));

        Linear { weights, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        let wx = matmul(&x, &self.weights);
        add(&wx, &self.bias)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![Rc::clone(&self.weights), Rc::clone(&self.bias)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ======
// Conv2d
// ======

pub struct Conv2d {
    pub weights: Rc<RefCell<Tensor>>,
    pub bias: Rc<RefCell<Tensor>>,
    stride: usize,
    padding: usize
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let weights = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![out_channels, in_channels, kernel_size, kernel_size], 0.01), true)));
        let bias = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![out_channels], 0.0), true)));

        Conv2d { weights, bias, stride, padding }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        conv2d(&x, &self.weights, Some(&self.bias), self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![Rc::clone(&self.weights), Rc::clone(&self.bias)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// =========
// MaxPool2d
// =========

pub struct MaxPool2d {
    kernel_size: usize,
    stride: usize,
    padding: usize
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        //let stride
        let stride = stride.unwrap_or(1);
        MaxPool2d { kernel_size, stride, padding }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        maxpool2d(&input, self.kernel_size, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// =======
// Flatten
// =======

pub struct Flatten {}

impl Flatten {
    pub fn new() -> Self {
        Flatten {}
    }
}

impl Module for Flatten {
    fn forward(&self, input: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        flatten(&input)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ====
// ReLU
// ====

pub struct ReLU {}

impl ReLU {
    pub fn new() -> Self {
        ReLU {  }
    }
}

impl Module for ReLU {
    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        relu(&x)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ====
// Sigmoid
// ====

pub struct Sigmoid {}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {  }
    }
}

impl Module for Sigmoid {
    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        sigmoid(&x)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ====
// Tanh
// ====

pub struct Tanh {}

impl Tanh {
    pub fn new() -> Self {
        Tanh {  }
    }
}

impl Module for Tanh {
    fn forward(&self, x: Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
        tanh(&x)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}