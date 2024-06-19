use ndarray::{iter::AxisIter, prelude::*, stack};
use ndarray_linalg::{error::LinalgError, Inverse};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

// fn random_ssm<B:Backend>(device: &B::Device, N: usize) -> (Tensor<B,2>, Tensor<B,2>, Tensor<B,2>) {
//     let A = Tensor::random([N,N], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
//     let B = Tensor::random([N,N], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
//     let C = Tensor::random([N,N], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
//     (A, B, C)
// }

// fn discretize<B:Backend>(A: Tensor<B,2>, B: Tensor<B,2>, C: Tensor<B,2>, step: f32) -> (Tensor<B,2>, Tensor<B,2>, Tensor<B,2>) {
//     let I = Tensor::eye(A.dims()[0], &A.device());
//     let BL = (I - A.mul_scalar(step / 2.0)).inv();
//     // I = np.eye(A.shape[0])
//     // BL = inv(I - (step / 2.0) * A)
//     // Ab = BL @ (I + (step / 2.0) * A)
//     // Bb = (BL * step) @ B
//     // return Ab, Bb, C
// }

fn random_SSM(N: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let A = Array::random((N,N), Uniform::new(0.0, 1.0));
    let B = Array::random((N,1), Uniform::new(0.0, 1.0));
    let C = Array::random((1,N), Uniform::new(0.0, 1.0));
    (A, B, C)
}

// Discrete-time SSM: The Recurrent Representation

// TODO: go thruogh and check on allocations during arithmetic
fn discretize(A: Array2<f32>, B: Array2<f32>, C: Array2<f32>, step: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
// Result<(Array2<f32>, Array2<f32>, Array2<f32>), LinalgError> {
    let I: Array2<f32> = Array::eye(A.shape()[0]);
    // let BL = (&I - (step / 2.0) * &A).inv()?;
    let BL = (&I - (step / 2.0) * &A).inv().unwrap();
    let Ab = BL.dot(&(&I + (step / 2.0) * &A));
    let Bb = (step * BL).dot(&B);
    // Ok((Ab, Bb, C))
    (Ab, Bb, C)
}

// Step/scan function

fn scan_SSM(Ab: Array2<f32>, Bb: Array2<f32>, Cb: Array2<f32>, u: Array2<f32>, x0: Array1<f32>) -> (Array1<f32>, Array2<f32>) {
    let step = |x_k_1: Array1<f32>, u_k: Array1<f32>| {
        let x_k = Ab.dot(&x_k_1) + Bb.dot(&u_k);
        let y_k = Cb.dot(&x_k);
        (x_k, y_k)
    };
    scan(step, x0, u.axis_iter(Axis(0)))
}

fn run_SSM(A: Array2<f32>, B: Array2<f32>, C: Array2<f32>, u: Array1<f32>) -> (Array1<f32>, Array2<f32>) {
    let L = u.shape()[0] as f32;
    let N = A.shape()[0];
    let (Ab, Bb, Cb) = discretize(A, B, C, 1.0 / L);
    let x0 = Array::zeros((N,));
    scan_SSM(Ab, Bb, Cb, u.insert_axis(Axis(1)), x0)
}

fn example_mass(k: f32, b: f32, m: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let A = array![[0.0, 1.0], [-k / m, -b / m]];
    let B = array![[0.0], [1.0 / m]];
    let C = array![[1.0, 0.0]];
    (A, B, C)
}

fn example_force(t: f32) -> f32 {
    let x = (10.0 * t).sin();
    if x > 0.5 {
        x
    } else {
        0.0
    }
}

pub fn example_ssm() {
    let ssm = example_mass(40.0, 5.0, 1.0);
    let L = 100;
    let step = 1.0 / L as f32;
    let ks = Array::range(0.0, L as f32, 1.0);
    let u = ks.map(|x| example_force(x * step));
    println!("Force: {}", u);

    let (_, ys) = run_SSM(ssm.0, ssm.1, ssm.2, u);
    println!("Position: {}", ys.remove_axis(Axis(1)));
}

// -- convolution -- //

// fn K_conv(Ab: Array2<f32>, Bb: Array2<f32>, Cb: Array2<f32>, L: u32) -> Array2<f32> {

// }



// -- utils -- //


// ported from psudeocode at https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
fn scan<F>(f: F, init: Array1<f32>, xs: AxisIter<'_, f32, Dim<[usize; 1]>>) -> (Array1<f32>, Array2<f32>)
where F:Fn(Array1<f32>, Array1<f32>) -> (Array1<f32>, Array1<f32>) {
    // let mut ys: Vec<ndarray::ArrayBase<ndarray::ViewRepr<&_>, _>> = Vec::new();
    let mut ys = Vec::new();
    let mut carry = init;
    let mut y;
    for x in xs {
        (carry, y) = f(carry, x.to_owned());
        ys.push(y);
    }
    // let yss = ys.into_iter().map(|y| y.view()).collect::<Vec<_>>();
    let temp = ys.iter().map(|y| y.view()).collect::<Vec<_>>();
    let yyy: &[ndarray::ArrayBase<ndarray::ViewRepr<&_>, _>] = temp.as_slice();
    let ss = stack(Axis(0), yyy).unwrap();
    (carry, ss)
}
