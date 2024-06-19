This post is my process of going through The Annotated S4 and porting the code to Rust block by block.

I'll be using the Burn library and anything else needed that I discover along the way.
The code will follow the original blog post and associate github repo closely, so the rust
linter will complain a lot about capitalization.

To add dependencies:
```
apt-get install libopenblas-dev
cargo add burn ndarray ndarray-rand
cargo add ndarray-linalg --features openblas-system
```

TODO: come back to the imports
``` rust
use burn::prelude::*;
Backend::seed(1);
```

Starting out with the matrices A,B,C intialized randomly.

``` rust
pub fn random_ssm<B:Backend>(device: &B::Device, n: usize) -> (Tensor<B,2>, Tensor<B,2>, Tensor<B,2>) {
    let a = Tensor::random([n,n], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
    let b = Tensor::random([n,n], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
    let c = Tensor::random([n,n], burn::tensor::Distribution::Uniform(0.0, 1.0), device);
    (a, b, c)
}
```

