use cfg_if::cfg_if;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::info;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPrecision {
    Fp32,
    Fp16,
    Int8,
}

pub trait TensorExecutor: Send + Sync {
    fn gemm(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
        seq_len: usize,
        head_dim: usize,
        out: &mut [f32],
    );
    fn layer_norm(&self, data: &mut [f32], hidden_size: usize, eps: f32);
    fn softmax(&self, data: &mut [f32]);
    fn non_temporal_store(&self, src: &[f32], dst: &mut [f32]);
    fn precision(&self) -> TensorPrecision;
    fn threads(&self) -> usize;
    fn l2_distance_squared(&self, lhs: &[f32], rhs: &[f32], dim: usize, out: &mut [f32]);
}

pub type TensorExecutorHandle = Arc<dyn TensorExecutor>;

#[derive(Debug, Clone)]
pub struct TensorHardwareHints {
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub prefers_large_pages: bool,
}

impl Default for TensorHardwareHints {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get().max(1),
            total_memory_gb: 4.0,
            prefers_large_pages: false,
        }
    }
}

#[derive(Debug, Clone)]
struct TileShape {
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Debug, Clone, Copy)]
struct CpuFeatures {
    avx512: bool,
    avx2: bool,
    fma: bool,
    neon: bool,
    sve: bool,
}

impl CpuFeatures {
    fn detect() -> Self {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                Self {
                    avx512: std::arch::is_x86_feature_detected!("avx512f"),
                    avx2: std::arch::is_x86_feature_detected!("avx2"),
                    fma: std::arch::is_x86_feature_detected!("fma"),
                    neon: false,
                    sve: false,
                }
            } else if #[cfg(any(target_arch = "aarch64", target_arch = "arm"))] {
                let neon = cfg!(any(target_feature = "neon"));
                let sve = cfg!(target_feature = "sve");
                Self {
                    avx512: false,
                    avx2: false,
                    fma: false,
                    neon,
                    sve,
                }
            } else {
                Self {
                    avx512: false,
                    avx2: false,
                    fma: false,
                    neon: false,
                    sve: false,
                }
            }
        }
    }
}

pub struct CpuTensorExecutor {
    hints: TensorHardwareHints,
    features: CpuFeatures,
    tile: TileShape,
    precision: TensorPrecision,
}

impl CpuTensorExecutor {
    pub fn new(hints: TensorHardwareHints) -> Self {
        let features = CpuFeatures::detect();
        let precision = if features.avx512 {
            TensorPrecision::Fp16
        } else if features.avx2 {
            TensorPrecision::Fp32
        } else {
            TensorPrecision::Fp32
        };
        let tile = Self::choose_tile(&hints, &features);
        info!(
            target: "w1z4rdv1510n::tensor",
            cpu_cores = hints.cpu_cores,
            total_memory_gb = hints.total_memory_gb,
            avx2 = features.avx2,
            avx512 = features.avx512,
            neon = features.neon,
            sve = features.sve,
            tile_m = tile.m,
            tile_n = tile.n,
            tile_k = tile.k,
            precision = ?precision,
            "cpu tensor backend configured"
        );
        Self {
            hints,
            features,
            tile,
            precision,
        }
    }

    pub fn handle(hints: TensorHardwareHints) -> TensorExecutorHandle {
        Arc::new(Self::new(hints))
    }

    fn choose_tile(hints: &TensorHardwareHints, features: &CpuFeatures) -> TileShape {
        let core_factor = hints.cpu_cores.clamp(1, 64);
        let base = if features.avx512 {
            256
        } else if features.avx2 {
            192
        } else {
            96
        };
        let m = (base * core_factor / 8).clamp(32, 512);
        let n = (base * 2).clamp(64, 512);
        let k = if features.avx512 { 64 } else { 48 };
        TileShape { m, n, k }
    }

    #[inline(always)]
    fn fmadd_row(&self, acc: &mut [f32], scalar: f32, rhs: &[f32]) {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if self.features.avx2 {
                    unsafe { fmadd_row_avx(acc, scalar, rhs, self.features.fma); }
                    return;
                }
            } else if #[cfg(target_arch = "aarch64")] {
                if self.features.neon {
                    unsafe { fmadd_row_neon(acc, scalar, rhs); }
                    return;
                }
            }
        }
        for (dst, src) in acc.iter_mut().zip(rhs.iter()) {
            *dst += scalar * *src;
        }
    }

    fn gemm_block(
        &self,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        row_offset: usize,
        rows: usize,
        k: usize,
        n: usize,
    ) {
        for local_row in 0..rows {
            let row = row_offset + local_row;
            if row * k >= a.len() {
                break;
            }
            let a_row = &a[row * k..row * k + k];
            let c_row = &mut out[local_row * n..local_row * n + n];
            c_row.fill(0.0);
            let mut kk = 0;
            while kk < k {
                let kk_end = (kk + self.tile.k).min(k);
                for inner in kk..kk_end {
                    let scalar = a_row[inner];
                    let rhs = &b[inner * n..inner * n + n];
                    prefetch(rhs);
                    self.fmadd_row(c_row, scalar, rhs);
                }
                kk = kk_end;
            }
        }
    }

    fn transpose(&self, input: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(out.len(), rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = input[r * cols + c];
            }
        }
    }

    fn threads_for(&self, rows: usize) -> usize {
        let threads = if rows < self.hints.cpu_cores {
            rows.max(1)
        } else {
            self.hints.cpu_cores
        };
        threads.max(1)
    }
}

impl TensorExecutor for CpuTensorExecutor {
    fn gemm(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        assert_eq!(a.len(), m * k, "lhs shape mismatch");
        assert_eq!(b.len(), k * n, "rhs shape mismatch");
        assert_eq!(out.len(), m * n, "output shape mismatch");
        out.fill(0.0);
        let threads = self.threads_for(m);
        let rows_per_chunk = (m + threads - 1) / threads.max(1);
        out.par_chunks_mut(rows_per_chunk * n)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * rows_per_chunk;
                if row_start >= m {
                    return;
                }
                let rows = rows_per_chunk.min(m - row_start);
                self.gemm_block(a, b, chunk, row_start, rows, k, n);
            });
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
        seq_len: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        let head_stride = seq_len * head_dim;
        assert_eq!(q.len(), heads * head_stride);
        assert_eq!(k.len(), heads * head_stride);
        assert_eq!(v.len(), heads * head_stride);
        assert_eq!(out.len(), heads * head_stride);
        let mut k_transposed = vec![0.0f32; head_dim * seq_len];
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for head in 0..heads {
            let q_head = &q[head * head_stride..(head + 1) * head_stride];
            let k_head = &k[head * head_stride..(head + 1) * head_stride];
            let v_head = &v[head * head_stride..(head + 1) * head_stride];
            let out_head = &mut out[head * head_stride..(head + 1) * head_stride];
            self.transpose(k_head, seq_len, head_dim, &mut k_transposed);
            self.gemm(
                q_head,
                &k_transposed,
                &mut scores,
                seq_len,
                head_dim,
                seq_len,
            );
            let scale = (head_dim as f32).recip().sqrt();
            scores.iter_mut().for_each(|s| *s *= scale);
            for row in scores.chunks_mut(seq_len) {
                self.softmax(row);
            }
            self.gemm(&scores, v_head, out_head, seq_len, seq_len, head_dim);
        }
    }

    fn layer_norm(&self, data: &mut [f32], hidden_size: usize, eps: f32) {
        assert_eq!(data.len() % hidden_size, 0);
        data.par_chunks_mut(hidden_size).for_each(|row| {
            let mean = row.iter().copied().sum::<f32>() / hidden_size as f32;
            let variance = row
                .iter()
                .copied()
                .map(|x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f32>()
                / hidden_size as f32;
            let inv_std = 1.0f32 / (variance + eps).sqrt();
            for value in row {
                *value = (*value - mean) * inv_std;
            }
        });
    }

    fn softmax(&self, data: &mut [f32]) {
        if data.is_empty() {
            return;
        }
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in data.iter_mut() {
            *value = (*value - max).exp();
            sum += *value;
        }
        let sum = sum.max(1e-9);
        for value in data.iter_mut() {
            *value /= sum;
        }
    }

    fn non_temporal_store(&self, src: &[f32], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len());
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                if self.features.avx512 {
                    unsafe { stream_store_avx512(src, dst); return; }
                } else if self.features.avx2 {
                    unsafe { stream_store_avx2(src, dst); return; }
                }
            } else if #[cfg(target_arch = "aarch64")] {
                if self.features.neon {
                    unsafe { stream_store_neon(src, dst); return; }
                }
            }
        }
        dst.copy_from_slice(src);
    }

    fn precision(&self) -> TensorPrecision {
        self.precision
    }

    fn threads(&self) -> usize {
        self.hints.cpu_cores
    }

    fn l2_distance_squared(&self, lhs: &[f32], rhs: &[f32], dim: usize, out: &mut [f32]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(lhs.len(), out.len() * dim);
        if dim == 0 {
            out.fill(0.0);
            return;
        }
        out.par_iter_mut().enumerate().for_each(|(row, accum)| {
            let start = row * dim;
            let end = start + dim;
            let mut sum = 0.0f32;
            for idx in start..end {
                let diff = lhs[idx] - rhs[idx];
                sum += diff * diff;
            }
            *accum = sum;
        });
    }
}

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        #[inline(always)]
        unsafe fn fmadd_row_avx(acc: &mut [f32], scalar: f32, rhs: &[f32], has_fma: bool) {
            use std::arch::x86_64::*;
            let lanes = 8;
            let mut idx = 0;
            unsafe {
                let scalar_vec = _mm256_set1_ps(scalar);
                while idx + lanes <= acc.len() && idx + lanes <= rhs.len() {
                    let rhs_vec = _mm256_loadu_ps(rhs.as_ptr().add(idx));
                    let acc_vec = _mm256_loadu_ps(acc.as_ptr().add(idx));
                    let res = if has_fma {
                        _mm256_fmadd_ps(scalar_vec, rhs_vec, acc_vec)
                    } else {
                        let mul = _mm256_mul_ps(scalar_vec, rhs_vec);
                        _mm256_add_ps(mul, acc_vec)
                    };
                    _mm256_storeu_ps(acc.as_mut_ptr().add(idx), res);
                    idx += lanes;
                }
            }
            while idx < acc.len() && idx < rhs.len() {
                acc[idx] += scalar * rhs[idx];
                idx += 1;
            }
        }

        #[inline(always)]
        unsafe fn stream_store_avx2(src: &[f32], dst: &mut [f32]) {
            use std::arch::x86_64::*;
            let lanes = 8;
            let mut idx = 0;
            unsafe {
                while idx + lanes <= src.len() {
                    let vec = _mm256_loadu_ps(src.as_ptr().add(idx));
                    _mm256_stream_ps(dst.as_mut_ptr().add(idx), vec);
                    idx += lanes;
                }
            }
            while idx < src.len() {
                dst[idx] = src[idx];
                idx += 1;
            }
        }

        #[inline(always)]
        unsafe fn stream_store_avx512(src: &[f32], dst: &mut [f32]) {
            use std::arch::x86_64::*;
            let lanes = 16;
            let mut idx = 0;
            unsafe {
                while idx + lanes <= src.len() {
                    let vec = _mm512_loadu_ps(src.as_ptr().add(idx));
                    _mm512_stream_ps(dst.as_mut_ptr().add(idx), vec);
                    idx += lanes;
                }
            }
            while idx < src.len() {
                dst[idx] = src[idx];
                idx += 1;
            }
        }

        #[inline(always)]
        fn prefetch(slice: &[f32]) {
            use std::arch::x86_64::_mm_prefetch;
            unsafe {
                let ptr = slice.as_ptr();
                _mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }
    } else if #[cfg(target_arch = "aarch64")] {
        #[inline(always)]
        unsafe fn fmadd_row_neon(acc: &mut [f32], scalar: f32, rhs: &[f32]) {
            use std::arch::aarch64::*;
            let lanes = 4;
            let scalar_vec = vdupq_n_f32(scalar);
            let mut idx = 0;
            while idx + lanes <= acc.len() && idx + lanes <= rhs.len() {
                let rhs_vec = vld1q_f32(rhs.as_ptr().add(idx));
                let acc_vec = vld1q_f32(acc.as_ptr().add(idx));
                let res = vfmaq_f32(acc_vec, scalar_vec, rhs_vec);
                vst1q_f32(acc.as_mut_ptr().add(idx), res);
                idx += lanes;
            }
            while idx < acc.len() && idx < rhs.len() {
                acc[idx] += scalar * rhs[idx];
                idx += 1;
            }
        }

        #[inline(always)]
        unsafe fn stream_store_neon(src: &[f32], dst: &mut [f32]) {
            use std::arch::aarch64::*;
            let lanes = 4;
            let mut idx = 0;
            while idx + lanes <= src.len() {
                let vec = vld1q_f32(src.as_ptr().add(idx));
                vst1q_f32(dst.as_mut_ptr().add(idx), vec);
                idx += lanes;
            }
            while idx < src.len() {
                dst[idx] = src[idx];
                idx += 1;
            }
        }

        #[inline(always)]
        fn prefetch(_slice: &[f32]) {}
    } else {
        #[inline(always)]
        fn prefetch(_slice: &[f32]) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hints() -> TensorHardwareHints {
        TensorHardwareHints {
            cpu_cores: 4,
            total_memory_gb: 16.0,
            prefers_large_pages: false,
        }
    }

    #[test]
    fn softmax_normalizes() {
        let executor = CpuTensorExecutor::new(hints());
        let mut data = vec![1.0, 2.0, 3.0];
        executor.softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn gemm_produces_expected_shape() {
        let executor = CpuTensorExecutor::new(hints());
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 4];
        executor.gemm(&a, &b, &mut out, 2, 2, 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_zero_mean() {
        let executor = CpuTensorExecutor::new(hints());
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        executor.layer_norm(&mut data, 4, 1e-5);
        let mean = data.iter().copied().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-3);
    }
}
