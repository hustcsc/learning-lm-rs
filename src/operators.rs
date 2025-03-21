use std::vec;
use serde::de;
use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}
// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let mut offset = 0;
    while offset + w.size() <= x.size() {
        let _x_slice = x.slice(offset, w.shape());
        let mut _y_slice = y.slice(offset, w.shape());

        let _y_slice_data = unsafe {_y_slice.data_mut()};
        let _x_slice_data = _x_slice.data();

        let len = _y_slice_data.len();

        let denominator: f32 = (dot(&_x_slice, &_x_slice) / (len as f32) + epsilon).sqrt();
        for i in 0..len {
            _y_slice_data[i] = w.data()[i] * _x_slice_data[i] / denominator;
        }
        offset += w.size();
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for i in 0..len {
        _y[i] = _x[i] / (1.0 + (-_x[i]).exp()) * _y[i];
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");

    // a (m,k) b (n,k) c (m,n)
    assert_eq!(a.shape()[1], b.shape()[1], "矩阵a,b乘法的维度不匹配");  
    assert_eq!(c.shape()[0], a.shape()[0], "矩阵a,c乘法的维度不匹配");
    assert_eq!(c.shape()[1], b.shape()[0], "矩阵b,c乘法的维度不匹配");
    // let b_t = b.transpose();
    let c_data = unsafe { c.data_mut() };
    let a_data = a.data();
    let b_data = b.data();
    let m = a.shape()[0];
    let n = b.shape()[0];
    let k = a.shape()[1];
    // a (m,k) b (n,k) c (m,n)
    // 遍历 C 的每个元素
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;

            // 计算 A[i, :] * B[j, :]^T
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[j * k + l];
            }

            // 更新 C[i, j]
            let index = i * n + j;
            c_data[index] = alpha * sum + beta * c_data[index];
        }
    }
}


// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
pub fn matmul_transb_avx(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (i, j, k) = (a_shape[0], b_shape[0], a_shape[1]);

    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    if is_x86_feature_detected!("avx2") && k % 8 == 0 {
        unsafe {
            for x in 0..i {
                for y in 0..j {
                    let mut sum_vec = _mm256_setzero_ps();
                    for z in (0..k).step_by(8) {
                        let a_vec = _mm256_loadu_ps(_a.as_ptr().add(x * k + z));
                        let b_vec = _mm256_loadu_ps(_b.as_ptr().add(y * k + z));
                        sum_vec   = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                    }
                    let mut sum_arr = [0.0; 8];
                    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
                    let sum = sum_arr.iter().sum::<f32>();

                    _c[x * j + y] *= beta;
                    _c[x * j + y] += alpha * sum;
                }
            }
        }
    } else {
        for x in 0..i {
            for y in 0..j {
                let mut sum = 0.0;
                for z in 0..k {
                    sum += _a[x * k + z] * _b[y * k + z];
                }
                _c[x * j + y] *= beta;
                _c[x * j + y] += alpha * sum;
            }
        }
    }
}


// #[test]
// fn test_gather() {
//     // 创建一个示例 table 张量
//     let table = Tensor::<f32>::new(
//         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
//         &vec![2, 4],
//     );

//     // 创建一个示例 indices 张量
//     let indices = Tensor::<u32>::new(vec![1, 0], &vec![2]);

//     // 创建一个空的 y 张量，用于存储结果
//     let mut y = Tensor::<f32>::new(vec![0.0; 8], &vec![2, 4]);

//     // 调用 gather 函数
//     gather(&mut y, &indices, &table);

//     // 预期结果
//     let expected = Tensor::<f32>::new(
//         vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0],
//         &vec![2, 4],
//     );

//     // 验证结果是否符合预期
//     assert!(y.close_to(&expected, 1e-3));
// }
