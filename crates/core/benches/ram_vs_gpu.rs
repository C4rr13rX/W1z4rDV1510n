use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::Rng;
use w1z4rdv1510n::hardware::{BitOp, CpuRamOptimizedBackend, HardwareBackend, HardwareProfile};

fn baseline_and(lhs: &[u8], rhs: &[u8], out: &mut [u8]) {
    for i in 0..out.len() {
        out[i] = lhs[i] & rhs[i];
    }
}

fn cpu_ram_backend() -> CpuRamOptimizedBackend {
    let detected = HardwareProfile::detect();
    let profile = HardwareProfile {
        cpu_cores: detected.cpu_cores.max(8),
        total_memory_gb: detected.total_memory_gb.max(8.0),
        has_gpu: false,
        cluster_hint: false,
    };
    CpuRamOptimizedBackend::new_with_profile(7, profile)
}

fn bench_bitops(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitwise_ram_vs_naive");
    let sizes = [256usize * 1024, 512 * 1024, 1024 * 1024];
    let backend = cpu_ram_backend();
    for size in sizes {
        let mut lhs = vec![0u8; size];
        let mut rhs = vec![0u8; size];
        let mut out = vec![0u8; size];
        let mut rng = rand::thread_rng();
        lhs.iter_mut().for_each(|b| *b = rng.r#gen::<u8>());
        rhs.iter_mut().for_each(|b| *b = rng.r#gen::<u8>());

        group.bench_function(BenchmarkId::new("naive_and", size), |b| {
            b.iter(|| baseline_and(&lhs, &rhs, &mut out));
        });
        group.bench_function(BenchmarkId::new("cpu_ram_bulk_and", size), |b| {
            b.iter(|| {
                let mut buffers = [lhs.as_mut_slice(), rhs.as_mut_slice(), out.as_mut_slice()];
                backend
                    .bulk_bitop(BitOp::And, &mut buffers)
                    .expect("bulk bitop works");
            });
        });
    }
    group.finish();
}

criterion_group!(hardware_benches, bench_bitops);
criterion_main!(hardware_benches);
