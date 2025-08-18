use std::cell::RefCell;
use std::rc::Rc;
use ndarray::{ArrayD, Ix2};

#[derive(Clone)]
pub struct Tensor {
    pub data: ArrayD<f64>,
    pub grad: Option<ArrayD<f64>>,
    requires_grad: bool,
    grad_fn: Option<Rc<RefCell<dyn Fn(&mut Tensor) -> ()>>>, // Backward function
}

impl Tensor {
    pub fn new(data: ArrayD<f64>, requires_grad: bool) -> Self {
        Tensor {
            data,
            grad: None,
            requires_grad,
            grad_fn: None,
        }
    }

    pub fn backward(&mut self) {
        if let Some(grad_fn) = &self.grad_fn {
            let grad_fn_clone = Rc::clone(grad_fn);
            grad_fn_clone.borrow_mut()(self);
        }
    }
}

pub fn add(a: &Rc<RefCell<Tensor>>, b: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let result_data = &a.borrow().data + &b.borrow().data;
    let result = Rc::new(RefCell::new(Tensor::new(result_data, true)));

    if a.borrow().requires_grad || b.borrow().requires_grad {
        let a_clone = Rc::clone(a);
        let b_clone = Rc::clone(b);
        let result_clone = Rc::clone(&result);

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
                let shape = b_clone.borrow().data.raw_dim();
                b_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }
        
            *a_clone.borrow_mut().grad.as_mut().unwrap() += out.grad.as_ref().unwrap();
            *b_clone.borrow_mut().grad.as_mut().unwrap() += out.grad.as_ref().unwrap();
        })));        
    }

    result
}

pub fn matmul(a: &Rc<RefCell<Tensor>>, b: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    // Forward pass
    //let result_data = a.borrow().data.clone().dot(&b.borrow().data.clone()); // ndarray's dot product
    let a_data = a.borrow().data.clone().into_dimensionality::<Ix2>().expect("a is not 2D");
    let b_data = b.borrow().data.clone().into_dimensionality::<Ix2>().expect("b is not 2D");
    let result_data = a_data.dot(&b_data);
    let result = Rc::new(RefCell::new(Tensor::new(result_data.into_dyn(), true)));

    // Backward pass
    if a.borrow().requires_grad || b.borrow().requires_grad {
        let a_clone = Rc::clone(a);
        let b_clone = Rc::clone(b);
        let result_clone = Rc::clone(&result);

        result.borrow_mut().grad_fn = Some(Rc::new(RefCell::new(move |out: &mut Tensor| {
            if out.grad.is_none() {
                out.grad = Some(ArrayD::zeros(out.data.raw_dim()));
            }

            // Gradient for a: out.grad * b^T
            if a_clone.borrow().grad.is_none() {
                let shape = a_clone.borrow().data.raw_dim();
                a_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }
            let out_grad_2d = out.grad.as_ref().unwrap()
                .view()
                .into_dimensionality::<Ix2>()
                .expect("out.grad not 2D");
            let binding = b_clone.borrow();
            let b_data_2d = binding.data
                .view()
                .into_dimensionality::<Ix2>()
                .expect("b not 2D");

            let grad_a = out_grad_2d.dot(&b_data_2d.t());
            *a_clone.borrow_mut().grad.as_mut().unwrap() += &grad_a.into_dyn();

            // Gradient for b: a^T * out.grad
            if b_clone.borrow().grad.is_none() {
                let shape = b_clone.borrow().data.raw_dim();
                b_clone.borrow_mut().grad = Some(ArrayD::zeros(shape));
            }
            let binding = a_clone.borrow();
            let a_data_2d = binding.data
                .view()
                .into_dimensionality::<Ix2>()
                .expect("a not 2D");
            let out_grad_2d = out.grad.as_ref().unwrap()
                .view()
                .into_dimensionality::<Ix2>()
                .expect("out.grad not 2D");

            let grad_b = a_data_2d.t().dot(&out_grad_2d);
            *b_clone.borrow_mut().grad.as_mut().unwrap() += &grad_b.into_dyn();
        })));
    }

    result
}