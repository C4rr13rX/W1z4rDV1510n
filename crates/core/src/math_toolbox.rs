use std::f64::consts::PI;

#[derive(Debug, Clone, Default)]
pub struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl RunningStats {
    pub fn update(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        if self.count == 1 {
            self.mean = value;
            self.m2 = 0.0;
            self.min = value;
            self.max = value;
            return;
        }
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    pub fn variance(&self, ddof: usize) -> Option<f64> {
        if self.count <= ddof {
            return None;
        }
        Some(self.m2 / (self.count - ddof) as f64)
    }

    pub fn stddev(&self, ddof: usize) -> Option<f64> {
        self.variance(ddof).map(|val| val.sqrt())
    }

    pub fn min(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    pub fn max(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max)
        }
    }
}

pub fn mean(values: &[f64]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &val in values {
        if val.is_finite() {
            sum += val;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

pub fn variance(values: &[f64], ddof: f64) -> Option<f64> {
    let vals = finite_values(values);
    let n = vals.len();
    if n == 0 {
        return None;
    }
    let mean = mean(&vals)?;
    let mut sum = 0.0;
    for val in &vals {
        let diff = val - mean;
        sum += diff * diff;
    }
    let denom = (n as f64 - ddof).max(1.0);
    Some(sum / denom)
}

pub fn stddev(values: &[f64], ddof: f64) -> Option<f64> {
    variance(values, ddof).map(|val| val.sqrt())
}

pub fn min_max(values: &[f64]) -> Option<(f64, f64)> {
    let vals = finite_values(values);
    let mut iter = vals.into_iter();
    let first = iter.next()?;
    let mut min = first;
    let mut max = first;
    for val in iter {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }
    Some((min, max))
}

pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut vals = finite_values(values);
    if vals.is_empty() {
        return None;
    }
    let p = p.clamp(0.0, 1.0);
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    if n == 1 {
        return Some(vals[0]);
    }
    let idx = p * (n as f64 - 1.0);
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        return Some(vals[lower]);
    }
    let frac = idx - lower as f64;
    Some(vals[lower] * (1.0 - frac) + vals[upper] * frac)
}

pub fn median(values: &[f64]) -> Option<f64> {
    percentile(values, 0.5)
}

pub fn trimmed_mean(values: &[f64], trim_fraction: f64) -> Option<f64> {
    let mut vals = finite_values(values);
    if vals.is_empty() {
        return None;
    }
    let trim = trim_fraction.clamp(0.0, 0.49);
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    let trim_count = ((n as f64) * trim) as usize;
    let slice = &vals[trim_count..n.saturating_sub(trim_count)];
    mean(slice)
}

pub fn mad(values: &[f64]) -> Option<f64> {
    let med = median(values)?;
    let mut deviations = Vec::with_capacity(values.len());
    for val in values {
        if val.is_finite() {
            deviations.push((val - med).abs());
        }
    }
    median(&deviations)
}

pub fn skewness(values: &[f64]) -> Option<f64> {
    let vals = finite_values(values);
    if vals.len() < 3 {
        return None;
    }
    let mean = mean(&vals)?;
    let std = stddev(&vals, 1.0)?;
    if std <= 0.0 {
        return None;
    }
    let mut sum = 0.0;
    for val in &vals {
        let z = (val - mean) / std;
        sum += z * z * z;
    }
    Some(sum / vals.len() as f64)
}

pub fn kurtosis(values: &[f64]) -> Option<f64> {
    let vals = finite_values(values);
    if vals.len() < 4 {
        return None;
    }
    let mean = mean(&vals)?;
    let std = stddev(&vals, 1.0)?;
    if std <= 0.0 {
        return None;
    }
    let mut sum = 0.0;
    for val in &vals {
        let z = (val - mean) / std;
        sum += z.powi(4);
    }
    Some(sum / vals.len() as f64 - 3.0)
}

pub fn zscore(values: &[f64]) -> Option<Vec<f64>> {
    let mean = mean(values)?;
    let std = stddev(values, 0.0)?;
    if std <= 0.0 {
        return None;
    }
    let mut out = Vec::with_capacity(values.len());
    for val in values {
        if val.is_finite() {
            out.push((val - mean) / std);
        } else {
            out.push(0.0);
        }
    }
    Some(out)
}

pub fn ewma(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(values.len());
    let mut last = values[0];
    out.push(last);
    for &val in values.iter().skip(1) {
        if val.is_finite() {
            last = alpha * val + (1.0 - alpha) * last;
        }
        out.push(last);
    }
    out
}

pub fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    if values.is_empty() || window == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in 0..values.len() {
        let val = values[idx];
        if val.is_finite() {
            sum += val;
            count += 1;
        }
        if idx >= window {
            let old = values[idx - window];
            if old.is_finite() {
                sum -= old;
                count = count.saturating_sub(1);
            }
        }
        if count == 0 {
            out.push(0.0);
        } else {
            out.push(sum / count as f64);
        }
    }
    out
}

pub fn autocorrelation(values: &[f64], lag: usize) -> Option<f64> {
    if lag == 0 || values.len() <= lag {
        return None;
    }
    let mean = mean(values)?;
    let mut num = 0.0;
    let mut denom = 0.0;
    for idx in 0..values.len() - lag {
        let a = values[idx] - mean;
        let b = values[idx + lag] - mean;
        num += a * b;
    }
    for val in values {
        let diff = val - mean;
        denom += diff * diff;
    }
    if denom <= 0.0 {
        None
    } else {
        Some(num / denom)
    }
}

pub fn cross_correlation(a: &[f64], b: &[f64], lag: usize) -> Option<f64> {
    if a.len() <= lag || b.len() <= lag {
        return None;
    }
    let len = a.len().min(b.len()) - lag;
    if len == 0 {
        return None;
    }
    let mean_a = mean(a)?;
    let mean_b = mean(b)?;
    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for idx in 0..len {
        let av = a[idx] - mean_a;
        let bv = b[idx + lag] - mean_b;
        num += av * bv;
        denom_a += av * av;
        denom_b += bv * bv;
    }
    let denom = (denom_a.sqrt() * denom_b.sqrt()).max(1e-9);
    Some((num / denom).clamp(-1.0, 1.0))
}

pub fn normalize_probs(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut sum = 0.0;
    let mut out = Vec::with_capacity(values.len());
    for &val in values {
        let v = if val.is_finite() && val > 0.0 { val } else { 0.0 };
        sum += v;
        out.push(v);
    }
    if sum <= 0.0 {
        let uniform = 1.0 / values.len() as f64;
        return vec![uniform; values.len()];
    }
    for val in &mut out {
        *val /= sum;
    }
    out
}

pub fn logsumexp(values: &[f64]) -> Option<f64> {
    let vals = finite_values(values);
    if vals.is_empty() {
        return None;
    }
    let max = vals
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));
    let mut sum = 0.0;
    for val in vals {
        sum += (val - max).exp();
    }
    Some(max + sum.ln())
}

pub fn softmax(values: &[f64]) -> Vec<f64> {
    let Some(lse) = logsumexp(values) else {
        return vec![0.0; values.len()];
    };
    values
        .iter()
        .map(|val| if val.is_finite() { (val - lse).exp() } else { 0.0 })
        .collect()
}

pub fn entropy(values: &[f64]) -> f64 {
    let probs = normalize_probs(values);
    let mut sum = 0.0;
    for p in probs {
        let p = p.clamp(1e-9, 1.0);
        sum -= p * p.ln();
    }
    sum
}

pub fn cross_entropy(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.is_empty() || q.is_empty() {
        return None;
    }
    let p = normalize_probs(p);
    let q = normalize_probs(q);
    if p.len() != q.len() {
        return None;
    }
    let mut sum = 0.0;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let qi = qi.clamp(1e-9, 1.0);
        sum -= pi * qi.ln();
    }
    Some(sum)
}

pub fn kl_divergence(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.is_empty() || q.is_empty() {
        return None;
    }
    let p = normalize_probs(p);
    let q = normalize_probs(q);
    if p.len() != q.len() {
        return None;
    }
    let mut sum = 0.0;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let pi = pi.clamp(1e-9, 1.0);
        let qi = qi.clamp(1e-9, 1.0);
        sum += pi * (pi / qi).ln();
    }
    Some(sum)
}

pub fn js_divergence(p: &[f64], q: &[f64]) -> Option<f64> {
    if p.is_empty() || q.is_empty() {
        return None;
    }
    let p = normalize_probs(p);
    let q = normalize_probs(q);
    if p.len() != q.len() {
        return None;
    }
    let mut m = Vec::with_capacity(p.len());
    for (pi, qi) in p.iter().zip(q.iter()) {
        m.push(0.5 * (pi + qi));
    }
    let kl_pm = kl_divergence(&p, &m)?;
    let kl_qm = kl_divergence(&q, &m)?;
    Some(0.5 * (kl_pm + kl_qm))
}

pub fn bayes_update(prior: &[f64], likelihood: &[f64]) -> Option<Vec<f64>> {
    if prior.is_empty() || likelihood.is_empty() {
        return None;
    }
    if prior.len() != likelihood.len() {
        return None;
    }
    let mut posterior = Vec::with_capacity(prior.len());
    for (p, l) in prior.iter().zip(likelihood.iter()) {
        let value = if p.is_finite() && l.is_finite() {
            (p.max(0.0)) * (l.max(0.0))
        } else {
            0.0
        };
        posterior.push(value);
    }
    Some(normalize_probs(&posterior))
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn dirichlet_mean(alpha: &[f64]) -> Option<Vec<f64>> {
    if alpha.is_empty() {
        return None;
    }
    let sum: f64 = alpha.iter().filter(|v| v.is_finite()).sum();
    if sum <= 0.0 {
        return None;
    }
    let mut out = Vec::with_capacity(alpha.len());
    for &val in alpha {
        out.push((val / sum).max(0.0));
    }
    Some(out)
}

pub fn dirichlet_entropy(alpha: &[f64]) -> Option<f64> {
    if alpha.is_empty() {
        return None;
    }
    let sum: f64 = alpha.iter().filter(|v| v.is_finite()).sum();
    if sum <= 0.0 {
        return None;
    }
    let mut ln_gamma_sum = 0.0;
    let mut psi_sum = 0.0;
    for &val in alpha {
        if val <= 0.0 {
            return None;
        }
        ln_gamma_sum += ln_gamma(val);
        psi_sum += (val - 1.0) * digamma(val);
    }
    let ln_gamma_total = ln_gamma(sum);
    let entropy = ln_gamma_sum - ln_gamma_total + (sum - alpha.len() as f64) * digamma(sum) - psi_sum;
    Some(entropy)
}

pub fn beta_mean(alpha: f64, beta: f64) -> Option<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        return None;
    }
    Some(alpha / (alpha + beta))
}

pub fn beta_variance(alpha: f64, beta: f64) -> Option<f64> {
    if alpha <= 0.0 || beta <= 0.0 {
        return None;
    }
    let sum = alpha + beta;
    Some((alpha * beta) / (sum * sum * (sum + 1.0)))
}

pub fn dot(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        sum += av * bv;
    }
    Some(sum)
}

pub fn l1_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).sum()
}

pub fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
    let dot = dot(a, b)?;
    let denom = l2_norm(a) * l2_norm(b);
    if denom <= 0.0 {
        None
    } else {
        Some((dot / denom).clamp(-1.0, 1.0))
    }
}

pub fn euclidean_distance(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let diff = av - bv;
        sum += diff * diff;
    }
    Some(sum.sqrt())
}

pub fn add_vectors(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = Vec::with_capacity(a.len());
    for (av, bv) in a.iter().zip(b.iter()) {
        out.push(av + bv);
    }
    Some(out)
}

pub fn sub_vectors(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = Vec::with_capacity(a.len());
    for (av, bv) in a.iter().zip(b.iter()) {
        out.push(av - bv);
    }
    Some(out)
}

pub fn scale_vector(values: &[f64], scale: f64) -> Vec<f64> {
    values.iter().map(|v| v * scale).collect()
}

pub fn hadamard(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = Vec::with_capacity(a.len());
    for (av, bv) in a.iter().zip(b.iter()) {
        out.push(av * bv);
    }
    Some(out)
}

pub fn normalize_vector(values: &[f64]) -> Vec<f64> {
    let norm = l2_norm(values);
    if norm <= 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| v / norm).collect()
}

pub fn transpose(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if matrix.is_empty() {
        return None;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    if cols == 0 {
        return None;
    }
    let mut out = vec![vec![0.0; rows]; cols];
    for (r, row) in matrix.iter().enumerate() {
        if row.len() != cols {
            return None;
        }
        for (c, val) in row.iter().enumerate() {
            out[c][r] = *val;
        }
    }
    Some(out)
}

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_rows = b.len();
    let b_cols = b[0].len();
    if a_cols == 0 || b_cols == 0 {
        return None;
    }
    if a_cols != b_rows {
        return None;
    }
    for row in a {
        if row.len() != a_cols {
            return None;
        }
    }
    for row in b {
        if row.len() != b_cols {
            return None;
        }
    }
    let mut out = vec![vec![0.0; b_cols]; a_rows];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let aik = a[i][k];
            for j in 0..b_cols {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    Some(out)
}

pub fn matrix_vec_mul(matrix: &[Vec<f64>], vector: &[f64]) -> Option<Vec<f64>> {
    if matrix.is_empty() || vector.is_empty() {
        return None;
    }
    let cols = matrix[0].len();
    if cols != vector.len() {
        return None;
    }
    let mut out = Vec::with_capacity(matrix.len());
    for row in matrix {
        if row.len() != cols {
            return None;
        }
        let mut sum = 0.0;
        for (val, vec) in row.iter().zip(vector.iter()) {
            sum += val * vec;
        }
        out.push(sum);
    }
    Some(out)
}

pub fn outer_product(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(a.len());
    for av in a {
        let mut row = Vec::with_capacity(b.len());
        for bv in b {
            row.push(av * bv);
        }
        out.push(row);
    }
    out
}

pub fn trace(matrix: &[Vec<f64>]) -> Option<f64> {
    if matrix.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for (idx, row) in matrix.iter().enumerate() {
        if idx >= row.len() {
            return None;
        }
        sum += row[idx];
    }
    Some(sum)
}

pub fn covariance_matrix(samples: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if samples.is_empty() {
        return None;
    }
    let dims = samples[0].len();
    if dims == 0 {
        return None;
    }
    for row in samples {
        if row.len() != dims {
            return None;
        }
    }
    let mut means = vec![0.0; dims];
    let mut count = 0.0;
    for row in samples {
        for (idx, val) in row.iter().enumerate() {
            means[idx] += val;
        }
        count += 1.0;
    }
    if count <= 1.0 {
        return None;
    }
    for mean in &mut means {
        *mean /= count;
    }
    let mut cov = vec![vec![0.0; dims]; dims];
    for row in samples {
        for i in 0..dims {
            for j in i..dims {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    let denom = (count - 1.0).max(1.0);
    for i in 0..dims {
        for j in i..dims {
            cov[i][j] /= denom;
            cov[j][i] = cov[i][j];
        }
    }
    Some(cov)
}

pub fn correlation_matrix(samples: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let cov = covariance_matrix(samples)?;
    let dims = cov.len();
    let mut stds = vec![0.0; dims];
    for i in 0..dims {
        stds[i] = cov[i][i].max(0.0).sqrt();
    }
    let mut corr = vec![vec![0.0; dims]; dims];
    for i in 0..dims {
        for j in 0..dims {
            let denom = (stds[i] * stds[j]).max(1e-9);
            corr[i][j] = (cov[i][j] / denom).clamp(-1.0, 1.0);
        }
    }
    Some(corr)
}

pub fn pearson_corr(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mean_a = mean(a)?;
    let mean_b = mean(b)?;
    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let da = av - mean_a;
        let db = bv - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    let denom = (denom_a.sqrt() * denom_b.sqrt()).max(1e-9);
    Some((num / denom).clamp(-1.0, 1.0))
}

pub fn rmse(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        let diff = a - p;
        sum += diff * diff;
    }
    Some((sum / actual.len() as f64).sqrt())
}

pub fn mae(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        sum += (a - p).abs();
    }
    Some(sum / actual.len() as f64)
}

pub fn mape(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        if a.abs() < 1e-9 {
            continue;
        }
        sum += ((a - p) / a).abs();
        count += 1.0;
    }
    if count == 0.0 {
        None
    } else {
        Some(sum / count)
    }
}

pub fn r2_score(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    let mean = mean(actual)?;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        let diff = a - p;
        ss_res += diff * diff;
        let diff_mean = a - mean;
        ss_tot += diff_mean * diff_mean;
    }
    if ss_tot <= 0.0 {
        return None;
    }
    Some(1.0 - ss_res / ss_tot)
}

pub fn circular_mean(values: &[f64]) -> Option<f64> {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    let mut count = 0usize;
    for &val in values {
        if val.is_finite() {
            sin_sum += val.sin();
            cos_sum += val.cos();
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    Some(sin_sum.atan2(cos_sum))
}

pub fn circular_variance(values: &[f64]) -> Option<f64> {
    let mut sin_sum = 0.0;
    let mut cos_sum = 0.0;
    let mut count = 0usize;
    for &val in values {
        if val.is_finite() {
            sin_sum += val.sin();
            cos_sum += val.cos();
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let r = ((sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / count as f64).clamp(0.0, 1.0);
    Some(1.0 - r)
}

#[derive(Debug, Clone)]
pub struct TensorBlock {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl TensorBlock {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Option<Self> {
        let expected = shape.iter().product::<usize>();
        if expected != data.len() {
            return None;
        }
        Some(Self { shape, data })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>();
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn index_flat(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut stride = 1usize;
        let mut idx = 0usize;
        for (dim, &index) in self.shape.iter().rev().zip(indices.iter().rev()) {
            if index >= *dim {
                return None;
            }
            idx += index * stride;
            stride *= *dim;
        }
        Some(idx)
    }

    pub fn index(&self, indices: &[usize]) -> Option<f64> {
        let flat = self.index_flat(indices)?;
        self.data.get(flat).copied()
    }

    pub fn set(&mut self, indices: &[usize], value: f64) -> Option<()> {
        let flat = self.index_flat(indices)?;
        if let Some(slot) = self.data.get_mut(flat) {
            *slot = value;
            Some(())
        } else {
            None
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Option<Self> {
        let expected = shape.iter().product::<usize>();
        if expected != self.data.len() {
            return None;
        }
        Some(Self {
            shape,
            data: self.data.clone(),
        })
    }

    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        let data = self.data.iter().map(|v| f(*v)).collect();
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.sum() / self.data.len() as f64)
        }
    }
}

fn finite_values(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|v| v.is_finite()).collect()
}

fn ln_gamma(z: f64) -> f64 {
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return PI.ln() - (PI * z).sin().ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = coeffs[0];
    for (i, coeff) in coeffs.iter().enumerate().skip(1) {
        x += coeff / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    result += x.ln() - 0.5 * inv - inv2 / 12.0 + inv4 / 120.0 - inv4 * inv2 / 252.0;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_and_probabilities_work() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert!((mean(&values).unwrap() - 2.5).abs() < 1e-6);
        assert!((variance(&values, 0.0).unwrap() - 1.25).abs() < 1e-6);
        let probs = softmax(&values);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(entropy(&probs) > 0.0);
    }

    #[test]
    fn tensor_indexing_roundtrip() {
        let mut tensor = TensorBlock::zeros(vec![2, 3]);
        tensor.set(&[1, 2], 0.9).expect("set");
        let value = tensor.index(&[1, 2]).expect("index");
        assert!((value - 0.9).abs() < 1e-6);
    }

    #[test]
    fn matrix_multiplication_works() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![2.0, 0.0], vec![1.0, 2.0]];
        let out = matmul(&a, &b).expect("matmul");
        assert_eq!(out[0][0], 4.0);
        assert_eq!(out[0][1], 4.0);
        assert_eq!(out[1][0], 10.0);
        assert_eq!(out[1][1], 8.0);
    }
}
