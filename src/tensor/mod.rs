use std::{cell::RefCell, collections::HashSet};
use std::rc::Rc;
use ndarray::{Array2, ArrayD, Axis, Ix2};
use std::f64::EPSILON; // To avoid log(0)

#[derive(Clone)]
pub struct Tensor {
    pub data: ArrayD<f64>,
    pub grad: Option<ArrayD<f64>>,
    requires_grad: bool,
    grad_fn: Option<Rc<RefCell<dyn Fn(&mut Tensor) -> ()>>>, // Backward function
    parents: Vec<Rc<RefCell<Tensor>>>
}

impl Tensor {
    pub fn new(data: ArrayD<f64>, requires_grad: bool) -> Self {
        Tensor {
            data,
            grad: None,
            requires_grad,
            grad_fn: None,
            parents: vec![]
        }
    }

    pub fn backward(&mut self) {
        let mut topo_order: Vec<Rc<RefCell<Tensor>>> = vec![];
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_topo(t: &Rc<RefCell<Tensor>>, topo: &mut Vec<Rc<RefCell<Tensor>>>, visited: &mut HashSet<usize>) {
            let addr = Rc::as_ptr(t) as usize;
            if visited.contains(&addr) { return; }
            visited.insert(addr);
            for p in &t.borrow().parents {
                build_topo(p, topo, visited);
            }
            topo.push(Rc::clone(t));
        }

        self.grad = Some(Array2::from_elem((1, 1), 1.0).into_dyn());

        build_topo(&Rc::new(RefCell::new(self.clone())), &mut topo_order, &mut visited);

        for node in topo_order.into_iter().rev() {
            let grad_fn_opt = { node.borrow().grad_fn.clone() };

            if let Some(grad_fn) = grad_fn_opt {
                grad_fn.borrow_mut()(&mut node.borrow_mut());
            }
        }
        /*println!("Running backward");
        if let Some(grad_fn) = &self.grad_fn {
            let grad_fn_clone = Rc::clone(grad_fn);
            grad_fn_clone.borrow_mut()(self);
        }*/
    }
}

// Addition
pub fn add(a: &Rc<RefCell<Tensor>>, b: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let result_data = a_data + b_data;
    let result = Rc::new(RefCell::new(Tensor::new(result_data.into_dyn(), true)));
    result.borrow_mut().parents = vec![Rc::clone(&a), Rc::clone(&b)];
    
    if a.borrow().requires_grad || b.borrow().requires_grad {
        let a_clone = Rc::clone(a);
        let b_clone = Rc::clone(b);

        result.borrow_mut().grad_fn = Some(Rc::new(RefCell::new(move |out: &mut Tensor| {
            if out.grad.is_none() {
                out.grad = Some(ArrayD::zeros(out.data.raw_dim()));
            }
        
            // Fix: Avoid borrowing conflicts by storing shape first
            if a_clone.borrow().grad.is_none() {
                let shape = a_clone.borrow().data.raw_dim(); // Immutable borrow ends
                a_clone.borrow_mut().grad = Some(ArrayD::zeros(shape)); // Mutable borrow starts
            }
        
            if b_clone.borrow().grad.is_none() {
                let shape = out.data.raw_dim(); // b_clone.borrow().data.raw_dim();
                b_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }
        
            /*a_clone.borrow_mut().grad.as_mut().unwrap() += out.grad.as_ref().unwrap();
            *b_clone.borrow_mut().grad.as_mut().unwrap() += out.grad.as_ref().unwrap(); */
            let a_dim = a_clone.borrow().data.raw_dim();
            let b_dim = b_clone.borrow().data.raw_dim();
            let out_grad = out.grad.as_ref().unwrap();

            // Update a gradient
            if let Some(a_broadcast) = out_grad.broadcast(a_dim) {
                *a_clone.borrow_mut().grad.as_mut().unwrap() += &a_broadcast;
            } else {
                panic!("Broadcast failed: {:?} to {:?}", out_grad.shape(), a_clone.borrow().data.shape());
            }

            // Update b gradient
            // If grad has an extra batch dimension, reduce it
            let b_shape = {
                let binding = b_clone.borrow();
                binding.data.shape().to_owned() // clone the shape so we don’t hold borrow
            };
            let reduced_grad = if out_grad.shape()[0] == b_shape[0] {
                out_grad
            } else if b_shape[0] == 1 {
                // Average along the batch dimension
                &out_grad.mean_axis(ndarray::Axis(0)).unwrap().insert_axis(ndarray::Axis(0))
            } else {
                panic!(
                    "Gradient shape {:?} does not match parameter shape {:?}",
                    out_grad.shape(),
                    b_clone.borrow().data.shape()
                );
            };
            if let Some(b_broadcast) = reduced_grad.broadcast(b_shape) {
                *b_clone.borrow_mut().grad.as_mut().unwrap() += &b_broadcast;
            } else {
                panic!("Broadcast failed: {:?} to {:?}", reduced_grad.shape(), b_clone.borrow().data.shape());
            }
        })));        
    }

    result
}

// Multiplication
pub fn matmul(a: &Rc<RefCell<Tensor>>, b: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    // Forward pass
    let a_data = a.borrow().data.clone().into_dimensionality::<Ix2>().expect("a is not 2D");
    let b_data = b.borrow().data.clone().into_dimensionality::<Ix2>().expect("b is not 2D");
    let result_data = a_data.dot(&b_data.t());
    let result = Rc::new(RefCell::new(Tensor::new(result_data.into_dyn(), true)));
    result.borrow_mut().parents = vec![Rc::clone(&a), Rc::clone(&b)];

    // Backward pass
    if a.borrow().requires_grad || b.borrow().requires_grad {
        let a_clone = Rc::clone(a);
        let b_clone = Rc::clone(b);

        result.borrow_mut().grad_fn = Some(Rc::new(RefCell::new(move |out: &mut Tensor| {
            if out.grad.is_none() {
                out.grad = Some(ArrayD::zeros(out.data.raw_dim()));
            }

            // Ensure grads exist
            if a_clone.borrow().grad.is_none() {
                let shape = a_clone.borrow().data.raw_dim();
                a_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }
            if b_clone.borrow().grad.is_none() {
                let shape = b_clone.borrow().data.raw_dim();
                b_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }

            // Gradient for a: out.grad * b^T
            let out_grad_2d = out.grad.as_ref().unwrap()
                .view()
                .into_dimensionality::<Ix2>()
                .expect("out.grad not 2D");
            let b_data_2d = {
                let binding = b_clone.borrow();
                binding.data.view().into_dimensionality::<Ix2>().expect("b not 2D").to_owned()
            };

            let grad_a = out_grad_2d.dot(&b_data_2d);
            *a_clone.borrow_mut().grad.as_mut().unwrap() += &grad_a.into_dyn();

            // Gradient for b: a^T * out.grad
            let a_data_2d = {
                let binding = a_clone.borrow();
                binding.data.view().into_dimensionality::<Ix2>().expect("a not 2D").to_owned()
            };
            let out_grad_2d = out.grad.as_ref().unwrap()
                .view()
                .into_dimensionality::<Ix2>()
                .expect("out.grad not 2D");

            let grad_b = a_data_2d.t().dot(&out_grad_2d);
            *b_clone.borrow_mut().grad.as_mut().unwrap() += &grad_b.t().into_dyn();
        })));
    }

    result
}

// ReLU
pub fn relu(x: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let x_data = x.borrow().data.clone();
    let result_data = x_data.mapv(|v| v.max(0.0));
    let result = Rc::new(RefCell::new(Tensor::new(result_data, true)));
    result.borrow_mut().parents = vec![Rc::clone(&x)];

    if x.borrow().requires_grad {
        let x_clone = Rc::clone(x);

        result.borrow_mut().grad_fn = Some(Rc::new(RefCell::new(move |out: &mut Tensor| {
            if out.grad.is_none() {
                out.grad = Some(ArrayD::zeros(out.data.raw_dim()));
            }
            if x_clone.borrow().grad.is_none() {
                let shape = x_clone.borrow().data.raw_dim();
                x_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }

            let grad_mask = x_clone.borrow().data.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            *x_clone.borrow_mut().grad.as_mut().unwrap() += &(grad_mask * out.grad.as_ref().unwrap());
        })));
    }

    result
}

// Sigmoid
pub fn sigmoid(x: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let x_data = x.borrow().data.clone();
    let result_data = x_data.mapv(|v| 1.0 / (1.0 + (-v).exp()));
    let result = Rc::new(RefCell::new(Tensor::new(result_data.clone(), true)));
    result.borrow_mut().parents = vec![Rc::clone(&x)];

    if x.borrow().requires_grad {
        let x_clone = Rc::clone(x);
        let result_clone = result_data; // store the forward result for backward

        result.borrow_mut().grad_fn = Some(Rc::new(RefCell::new(move |out: &mut Tensor| {
            if out.grad.is_none() {
                out.grad = Some(ArrayD::zeros(out.data.raw_dim()));
            }
            if x_clone.borrow().grad.is_none() {
                let shape = x_clone.borrow().data.raw_dim();
                x_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }

            let grad = &result_clone * &(1.0 - &result_clone) * out.grad.as_ref().unwrap();
            *x_clone.borrow_mut().grad.as_mut().unwrap() += &grad;
        })));
    }

    result
}

// BCE loss function
pub fn bce_loss(pred: &Rc<RefCell<Tensor>>, target: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let y_pred = pred.borrow().data.clone().into_dimensionality::<Ix2>().expect("y_pred is not 2D");
    let y_target = target.borrow().data.clone().into_dimensionality::<Ix2>().expect("y_target is not 2D");

    // Compute BCE loss: - (y*log(p) + (1-y)*log(1-p))
    let loss_data = -(y_target.t().dot(&y_pred.mapv(|v| (v + EPSILON).ln())) +
        &(1.0 - y_target) * &y_pred.mapv(|v| (1.0 - v + EPSILON).ln()));

    let mean_loss = loss_data.mean().unwrap();
    let result = Rc::new(RefCell::new(Tensor::new(ArrayD::from_elem(vec![1], mean_loss), true)));
    result.borrow_mut().parents = vec![Rc::clone(&pred)];

    if pred.borrow().requires_grad {
        let pred_clone = Rc::clone(pred);
        let target_clone = Rc::clone(target);
        
        result.borrow_mut().grad_fn = Some(Rc::new(RefCell::new(move |_: &mut Tensor| {
            let y_pred = pred_clone.borrow().data.clone().into_dimensionality::<Ix2>().expect("y_pred is not 1D");
            let y_target = target_clone.borrow().data.clone().into_dimensionality::<Ix2>().expect("y_pred is not 1D");
            
            if pred_clone.borrow().grad.is_none() {
                let shape = pred_clone.borrow().data.raw_dim();
                pred_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }

            // Compute grad
            let grad = (y_pred - &y_target) / (y_target.len() as f64);
            
            // Accumulate gradients in pred
            *pred_clone.borrow_mut().grad.as_mut().unwrap() += &grad;
        })));
    }

    result
}