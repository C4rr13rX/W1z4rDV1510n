use crate::config::{ExperimentalHardwareConfig, HardwareOverrides};
use crate::planner::{BackendPlanner, BackendPlannerHandle};
use crate::schema::{ParticleState, Population};
use crate::system::{
    LockedBuffer, NumaBindingGuard, allocate_large_page_buffer, allocate_numa_buffer,
    configure_numa_affinity,
};
use crate::tensor::{CpuTensorExecutor, TensorExecutorHandle, TensorHardwareHints};
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use cfg_if::cfg_if;
use num_cpus;
use parking_lot::Mutex;
#[cfg(feature = "gpu")]
use pollster::block_on;
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "gpu")]
#[cfg(feature = "gpu")]
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use tracing::{debug, info, warn};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HardwareBackendType {
    Auto,
    Cpu,
    MultiThreadedCpu,
    CpuRamOptimized,
    Gpu,
    Distributed,
    External,
    Experimental,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BitOp {
    And,
    Or,
    Xor,
}

pub trait StorageExecutor: Send + Sync {
    fn offload_scan(&self, _query: &str) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("storage offload not implemented")
    }

    fn offload_energy_eval(&self, _batch: &[Vec<u8>]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("storage offload not implemented")
    }
}

pub trait AnalogArray: Send + Sync {
    fn accumulate(&self, weights: &[f32], inputs: &[f32]) -> Vec<f32>;
}

impl Default for HardwareBackendType {
    fn default() -> Self {
        HardwareBackendType::Auto
    }
}

pub trait HardwareBackend: Send + Sync {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    );
    fn noise_source(&self) -> NoiseSourceHandle;

    fn bulk_bitop<'a>(&self, _op: BitOp, _buffers: &'a mut [&'a mut [u8]]) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "bulk bit operations not supported by this backend"
        ))
    }

    fn storage_executor(&self) -> Option<&dyn StorageExecutor> {
        None
    }

    fn analog_array(&self) -> Option<&dyn AnalogArray> {
        None
    }

    fn tensor_executor(&self) -> Option<TensorExecutorHandle> {
        None
    }

    fn planner(&self) -> Option<BackendPlannerHandle> {
        None
    }
}

pub struct CpuBackend {
    noise: NoiseSourceHandle,
}

impl CpuBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

impl HardwareBackend for CpuBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        for particle in &mut population.particles {
            func(particle);
        }
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }

    fn bulk_bitop<'a>(&self, op: BitOp, buffers: &'a mut [&'a mut [u8]]) -> anyhow::Result<()> {
        anyhow::ensure!(buffers.len() >= 3, "bulk bitop expects lhs, rhs, out");
        let len = buffers[0].len();
        anyhow::ensure!(
            buffers.iter().all(|buf| buf.len() == len),
            "buffers must match"
        );
        let (lhs, rhs, out) = split_buffers(buffers);

        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ))]
        unsafe {
            use std::arch::x86_64::*;
            let lanes = 16;
            let mut i = 0;
            while i + lanes <= len {
                let lhs_vec = _mm_loadu_si128(lhs.as_ptr().add(i) as *const __m128i);
                let rhs_vec = _mm_loadu_si128(rhs.as_ptr().add(i) as *const __m128i);
                let result = match op {
                    BitOp::And => _mm_and_si128(lhs_vec, rhs_vec),
                    BitOp::Or => _mm_or_si128(lhs_vec, rhs_vec),
                    BitOp::Xor => _mm_xor_si128(lhs_vec, rhs_vec),
                };
                _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, result);
                i += lanes;
            }
            while i < len {
                out[i] = match op {
                    BitOp::And => lhs[i] & rhs[i],
                    BitOp::Or => lhs[i] | rhs[i],
                    BitOp::Xor => lhs[i] ^ rhs[i],
                };
                i += 1;
            }
        }

        #[cfg(not(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        )))]
        {
            for i in 0..len {
                out[i] = match op {
                    BitOp::And => lhs[i] & rhs[i],
                    BitOp::Or => lhs[i] | rhs[i],
                    BitOp::Xor => lhs[i] ^ rhs[i],
                };
            }
        }
        Ok(())
    }
}

pub struct MultiThreadedCpuBackend {
    noise: NoiseSourceHandle,
}

impl MultiThreadedCpuBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

pub struct CpuRamOptimizedBackend {
    multi: MultiThreadedCpuBackend,
    tensor: TensorExecutorHandle,
    planner: BackendPlannerHandle,
    _numa_guard: Option<NumaBindingGuard>,
    _large_page: Option<LockedBuffer>,
    _numa_buffer: Option<LockedBuffer>,
}

impl CpuRamOptimizedBackend {
    fn new(seed: u64, profile: &HardwareProfile) -> Self {
        let hints = TensorHardwareHints {
            cpu_cores: profile.cpu_cores,
            total_memory_gb: profile.total_memory_gb,
            prefers_large_pages: true,
        };
        let tensor = CpuTensorExecutor::handle(hints);
        let planner = Arc::new(BackendPlanner::new(profile.clone(), false));
        let numa_guard = configure_numa_affinity(profile.cpu_cores);
        let large_page = allocate_large_page_buffer(8 * 1024 * 1024);
        let numa_buffer = allocate_numa_buffer(4 * 1024 * 1024);
        info!(
            target: "w1z4rdv1510n::hardware",
            large_page = large_page.is_some(),
            numa_buffer = numa_buffer.is_some(),
            "cpu+ram optimized backend initialized"
        );
        Self {
            multi: MultiThreadedCpuBackend::new(seed),
            tensor,
            planner,
            _numa_guard: numa_guard,
            _large_page: large_page,
            _numa_buffer: numa_buffer,
        }
    }

    pub fn new_with_profile(seed: u64, profile: HardwareProfile) -> Self {
        Self::new(seed, &profile)
    }
}

impl HardwareBackend for CpuRamOptimizedBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        self.multi.map_particles(population, func);
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        self.multi.noise_source()
    }

    fn bulk_bitop<'a>(&self, op: BitOp, buffers: &'a mut [&'a mut [u8]]) -> anyhow::Result<()> {
        self.multi.bulk_bitop(op, buffers)
    }

    fn tensor_executor(&self) -> Option<TensorExecutorHandle> {
        Some(Arc::clone(&self.tensor))
    }

    fn planner(&self) -> Option<BackendPlannerHandle> {
        Some(Arc::clone(&self.planner))
    }
}

impl HardwareBackend for MultiThreadedCpuBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        population
            .particles
            .par_iter_mut()
            .for_each(|particle| func(particle));
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }

    fn bulk_bitop<'a>(&self, op: BitOp, buffers: &'a mut [&'a mut [u8]]) -> anyhow::Result<()> {
        if buffers.len() < 3 {
            anyhow::bail!("bulk bitop expects at least 3 buffers (lhs, rhs, out)");
        }
        anyhow::ensure!(buffers.len() >= 3, "bulk bitop expects lhs, rhs, out");
        let len = buffers[0].len();
        anyhow::ensure!(
            buffers.iter().all(|buf| buf.len() == len),
            "buffers must match"
        );
        let (lhs, rhs, out) = split_buffers(buffers);
        let chunk = 1024;
        lhs.par_chunks(chunk)
            .zip(rhs.par_chunks(chunk))
            .zip(out.par_chunks_mut(chunk))
            .for_each(|((lhs, rhs), out)| {
                for i in 0..out.len() {
                    out[i] = match op {
                        BitOp::And => lhs[i] & rhs[i],
                        BitOp::Or => lhs[i] | rhs[i],
                        BitOp::Xor => lhs[i] ^ rhs[i],
                    };
                }
            });
        Ok(())
    }
}

fn split_buffers<'a>(buffers: &'a mut [&'a mut [u8]]) -> (&'a [u8], &'a [u8], &'a mut [u8]) {
    let (first, rest) = buffers.split_at_mut(1);
    let (second, rest) = rest.split_at_mut(1);
    let (third, _) = rest.split_at_mut(1);
    (&first[0][..], &second[0][..], &mut third[0][..])
}

#[cfg(feature = "gpu")]
pub struct GpuBackend {
    noise: NoiseSourceHandle,
    chunk_size: usize,
    engine: Arc<GpuBitOpEngine>,
}

#[cfg(feature = "gpu")]
impl GpuBackend {
    fn new(seed: u64, chunk_size: usize) -> anyhow::Result<Self> {
        Ok(Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
            chunk_size,
            engine: Arc::new(GpuBitOpEngine::new()?),
        })
    }
}

#[cfg(feature = "gpu")]
impl HardwareBackend for GpuBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        population
            .particles
            .par_chunks_mut(self.chunk_size)
            .for_each(|chunk| chunk.iter_mut().for_each(func));
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }

    fn bulk_bitop<'a>(&self, op: BitOp, buffers: &'a mut [&'a mut [u8]]) -> anyhow::Result<()> {
        anyhow::ensure!(buffers.len() >= 3, "bulk bitop expects lhs, rhs, out");
        let len = buffers[0].len();
        anyhow::ensure!(
            buffers.iter().all(|buf| buf.len() == len),
            "buffers must match"
        );
        let (lhs, rhs, out) = split_buffers(buffers);
        let lhs_words = bytes_to_words(lhs);
        let rhs_words = bytes_to_words(rhs);
        anyhow::ensure!(
            lhs_words.len() == rhs_words.len(),
            "lhs/rhs word lengths mismatch"
        );
        let mut result_words = vec![0u32; lhs_words.len()];
        self.engine
            .run(op, &lhs_words, &rhs_words, &mut result_words)?;
        words_to_bytes(&result_words, len, out);
        Ok(())
    }
}

#[cfg(feature = "gpu")]
const GPU_BITOP_SHADER: &str = r#"
struct Meta {
    len_words : u32,
    op_code : u32,
};

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> meta: Meta;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= meta.len_words) {
        return;
    }
    let a = lhs[idx];
    let b = rhs[idx];
    var result : u32;
    switch meta.op_code {
        case 0u {
            result = a & b;
        }
        case 1u {
            result = a | b;
        }
        default {
            result = a ^ b;
        }
    }
    out[idx] = result;
}
"#;

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BitOpUniform {
    len_words: u32,
    op_code: u32,
    padding0: u32,
    padding1: u32,
}

#[cfg(feature = "gpu")]
struct GpuBitOpEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuBitOpEngine {
    fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("failed to find compatible GPU adapter"))?;
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("w1z4rdv1510n_gpu_device"),
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("w1z4rdv1510n_bitop_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GPU_BITOP_SHADER)),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("w1z4rdv1510n_bitop_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("w1z4rdv1510n_bitop_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("w1z4rdv1510n_bitop_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_layout,
        })
    }

    fn run(&self, op: BitOp, lhs: &[u32], rhs: &[u32], out: &mut [u32]) -> anyhow::Result<()> {
        anyhow::ensure!(lhs.len() == rhs.len(), "lhs/rhs length mismatch");
        anyhow::ensure!(lhs.len() == out.len(), "output length mismatch");
        let bytes = lhs.len() * std::mem::size_of::<u32>();
        let lhs_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w1z4rdv1510n_lhs"),
                contents: cast_slice(lhs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let rhs_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w1z4rdv1510n_rhs"),
                contents: cast_slice(rhs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w1z4rdv1510n_out"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let meta = BitOpUniform {
            len_words: lhs.len() as u32,
            op_code: match op {
                BitOp::And => 0,
                BitOp::Or => 1,
                BitOp::Xor => 2,
            },
            padding0: 0,
            padding1: 0,
        };
        let meta_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("w1z4rdv1510n_meta"),
                contents: bytes_of(&meta),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("w1z4rdv1510n_bitop_bind_group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("w1z4rdv1510n_bitop_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("w1z4rdv1510n_bitop_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((lhs.len() as u32) + 127) / 128;
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w1z4rdv1510n_bitop_staging"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, bytes as u64);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        {
            let slice = staging.slice(..);
            block_on(slice.map_async(wgpu::MapMode::Read))
                .map_err(|_| anyhow::anyhow!("GPU map_async failed"))?;
            let view = slice.get_mapped_range();
            let gpu_words: &[u32] = cast_slice(&view);
            out.copy_from_slice(gpu_words);
            drop(view);
        }
        staging.unmap();
        Ok(())
    }
}

#[cfg(feature = "gpu")]
fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    let mut words = Vec::with_capacity((bytes.len() + 3) / 4);
    for chunk in bytes.chunks(4) {
        let mut data = [0u8; 4];
        data[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(data));
    }
    words
}

#[cfg(feature = "gpu")]
fn words_to_bytes(words: &[u32], len_bytes: usize, out: &mut [u8]) {
    for (word_idx, word) in words.iter().enumerate() {
        let start = word_idx * 4;
        if start >= len_bytes {
            break;
        }
        let end = (start + 4).min(len_bytes);
        let bytes = word.to_le_bytes();
        let slice_len = end - start;
        out[start..end].copy_from_slice(&bytes[..slice_len]);
    }
}

pub struct ExternalBackend {
    noise: NoiseSourceHandle,
}

impl ExternalBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

impl HardwareBackend for ExternalBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        for particle in &mut population.particles {
            func(particle);
        }
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

pub struct DistributedBackend {
    noise: NoiseSourceHandle,
    chunk_size: usize,
    nodes: usize,
}

impl DistributedBackend {
    fn new(seed: u64, chunk_size: usize, nodes: usize) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
            chunk_size: chunk_size.max(1),
            nodes: nodes.max(1),
        }
    }
}

impl HardwareBackend for DistributedBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        if population.particles.is_empty() {
            return;
        }
        let mut chunk = self.chunk_size.max(1);
        if self.nodes > 0 {
            let per_node = (population.particles.len() + self.nodes - 1) / self.nodes;
            if per_node > 0 {
                chunk = chunk.max(per_node);
            }
        }
        let mut segments = Vec::new();
        let mut start = 0;
        while start < population.particles.len() {
            let end = (start + chunk).min(population.particles.len());
            let mut cloned = Vec::with_capacity(end - start);
            cloned.extend_from_slice(&population.particles[start..end]);
            segments.push((start, cloned));
            start = end;
        }
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for (start_index, mut data) in segments {
                let func = func;
                handles.push(scope.spawn(move || {
                    for particle in data.iter_mut() {
                        func(particle);
                    }
                    (start_index, data)
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok((start_index, chunk_data)) => {
                        let end = start_index + chunk_data.len();
                        population.particles[start_index..end].clone_from_slice(&chunk_data);
                    }
                    Err(_) => {
                        warn!(
                            target: "w1z4rdv1510n::hardware",
                            "distributed worker thread panicked"
                        );
                    }
                }
            }
        });
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

#[cfg(feature = "experimental-hw")]
pub struct ExperimentalHardwareBackend {
    noise: NoiseSourceHandle,
    options: ExperimentalHardwareConfig,
    system: Mutex<System>,
    last_sample: Mutex<std::time::Instant>,
}

#[cfg(feature = "experimental-hw")]
impl ExperimentalHardwareBackend {
    fn new(seed: u64, options: ExperimentalHardwareConfig) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
            options,
            system: Mutex::new(System::new()),
            last_sample: Mutex::new(std::time::Instant::now()),
        }
    }

    fn maybe_sample(&self) {
        let interval = std::time::Duration::from_secs_f64(
            self.options.max_sample_interval_secs.max(0.1).min(10.0),
        );
        let mut last = self.last_sample.lock();
        if last.elapsed() < interval {
            return;
        }
        *last = std::time::Instant::now();
        let mut system = self.system.lock();
        system.refresh_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::new())
                .with_memory(MemoryRefreshKind::everything()),
        );
        let loads = system.load_average();
        let used_memory_gb = (system.used_memory() as f64 / 1_048_576.0).max(0.0);
        tracing::info!(
            target: "w1z4rdv1510n::hardware::experimental",
            used_memory_gb,
            load_avg_one = loads.one,
            load_avg_five = loads.five,
            perf_counters = self.options.use_performance_counters,
            "sampled hardware metrics"
        );
    }
}

#[cfg(feature = "experimental-hw")]
impl HardwareBackend for ExperimentalHardwareBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        self.maybe_sample();
        for particle in &mut population.particles {
            func(particle);
        }
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

pub type HardwareBackendHandle = Arc<dyn HardwareBackend>;

pub fn create_hardware_backend(
    kind: HardwareBackendType,
    seed: u64,
    experimental: &ExperimentalHardwareConfig,
    overrides: &HardwareOverrides,
) -> anyhow::Result<(HardwareBackendHandle, HardwareBackendType)> {
    let mut profile = HardwareProfile::detect();
    if let Some(max_threads) = overrides.max_threads {
        profile.cpu_cores = profile.cpu_cores.min(max_threads.max(1));
    }
    let mut resolved = if let HardwareBackendType::Auto = kind {
        recommend_backend(&profile)
    } else {
        kind
    };
    if matches!(resolved, HardwareBackendType::Gpu) && !profile.has_gpu {
        warn!(
            target: "w1z4rdv1510n::hardware",
            "GPU backend requested but no GPU detected; falling back to CpuRamOptimized"
        );
        resolved = HardwareBackendType::CpuRamOptimized;
    }
    if matches!(resolved, HardwareBackendType::Distributed) && !profile.cluster_hint {
        warn!(
            target: "w1z4rdv1510n::hardware",
            "Distributed backend requested without cluster hints; falling back to CpuRamOptimized"
        );
        resolved = HardwareBackendType::CpuRamOptimized;
    }
    if matches!(resolved, HardwareBackendType::Gpu) && !overrides.allow_gpu {
        warn!(
            target: "w1z4rdv1510n::hardware",
            "GPU backend requested but disabled via hardware_overrides; falling back to CpuRamOptimized"
        );
        resolved = HardwareBackendType::CpuRamOptimized;
    }
    if matches!(resolved, HardwareBackendType::Distributed) && !overrides.allow_distributed {
        warn!(
            target: "w1z4rdv1510n::hardware",
            "Distributed backend requested but disabled via hardware_overrides; falling back to CpuRamOptimized"
        );
        resolved = HardwareBackendType::CpuRamOptimized;
    }
    log_backend_selection(&resolved, &profile);
    let handle = match resolved {
        HardwareBackendType::Auto => {
            anyhow::bail!("resolved backend should not be Auto");
        }
        HardwareBackendType::Cpu => (
            Arc::new(CpuBackend::new(seed)) as HardwareBackendHandle,
            HardwareBackendType::Cpu,
        ),
        HardwareBackendType::MultiThreadedCpu => (
            Arc::new(MultiThreadedCpuBackend::new(seed)) as HardwareBackendHandle,
            HardwareBackendType::MultiThreadedCpu,
        ),
        HardwareBackendType::CpuRamOptimized => (
            Arc::new(CpuRamOptimizedBackend::new(seed, &profile)) as HardwareBackendHandle,
            HardwareBackendType::CpuRamOptimized,
        ),
        HardwareBackendType::Gpu => {
            cfg_if! {
                if #[cfg(feature = "gpu")] {
                    let chunk = gpu_chunk_size(profile.cpu_cores, profile.total_memory_gb);
                    (
                        Arc::new(GpuBackend::new(seed, chunk)?) as HardwareBackendHandle,
                        HardwareBackendType::Gpu,
                    )
                } else {
                    anyhow::bail!("GPU backend requires building with the `gpu` feature");
                }
            }
        }
        HardwareBackendType::Distributed => {
            let chunk = distributed_chunk_size(profile.cpu_cores, profile.total_memory_gb);
            (
                Arc::new(DistributedBackend::new(
                    seed,
                    chunk,
                    profile.cpu_cores.max(2),
                )) as HardwareBackendHandle,
                HardwareBackendType::Distributed,
            )
        }
        HardwareBackendType::External => (
            Arc::new(ExternalBackend::new(seed)) as HardwareBackendHandle,
            HardwareBackendType::External,
        ),
        HardwareBackendType::Experimental => create_experimental_backend(seed, experimental)?,
    };
    Ok(handle)
}

fn create_experimental_backend(
    seed: u64,
    config: &ExperimentalHardwareConfig,
) -> anyhow::Result<(HardwareBackendHandle, HardwareBackendType)> {
    anyhow::ensure!(
        config.enabled,
        "experimental hardware backend selected but experimental_hardware.enabled is false"
    );
    #[cfg(feature = "experimental-hw")]
    {
        Ok((
            Arc::new(ExperimentalHardwareBackend::new(seed, config.clone()))
                as HardwareBackendHandle,
            HardwareBackendType::Experimental,
        ))
    }
    #[cfg(not(feature = "experimental-hw"))]
    {
        let _ = seed;
        let _ = config;
        anyhow::bail!(
            "experimental hardware backend requires building with the `experimental-hw` feature"
        );
    }
}

pub trait NoiseSource: Send + Sync {
    fn random_float(&self) -> f64;
    fn random_int(&self, upper: u64) -> u64;
    fn entropy_status(&self) -> HashMap<String, String>;
}

pub type NoiseSourceHandle = Arc<dyn NoiseSource>;

struct SoftwareNoiseSource {
    rng: Mutex<StdRng>,
}

impl SoftwareNoiseSource {
    fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl NoiseSource for SoftwareNoiseSource {
    fn random_float(&self) -> f64 {
        self.rng.lock().r#gen()
    }

    fn random_int(&self, upper: u64) -> u64 {
        if upper == 0 {
            0
        } else {
            self.rng.lock().gen_range(0..upper)
        }
    }

    fn entropy_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("type".into(), "software_rng".into());
        status.insert("backing".into(), "StdRng".into());
        status
    }
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub has_gpu: bool,
    pub cluster_hint: bool,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        let mut system = System::new();
        system.refresh_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything()),
        );
        let cpu_cores = num_cpus::get().max(1);
        let total_memory_gb = (system.total_memory() as f64 / 1_048_576.0).max(0.5);
        let has_gpu_env =
            read_env_bool("SIMFUTURES_HAS_GPU") || read_env_bool("W1Z4RDV1510N_HAS_GPU");
        let cuda_visible = std::env::var("CUDA_VISIBLE_DEVICES")
            .map(|v| !v.trim().is_empty() && v.trim() != "-1")
            .unwrap_or(false);
        let has_gpu = has_gpu_env || cuda_visible;
        let cluster_hint = read_env_bool("SIMFUTURES_DISTRIBUTED")
            || read_env_bool("W1Z4RDV1510N_DISTRIBUTED")
            || std::env::var("SLURM_JOB_ID").is_ok();
        let profile = Self {
            cpu_cores,
            total_memory_gb,
            has_gpu,
            cluster_hint,
        };
        debug!(
            target: "w1z4rdv1510n::hardware",
            cpu_cores = profile.cpu_cores,
            memory_gb = profile.total_memory_gb,
            has_gpu = profile.has_gpu,
            cluster_hint = profile.cluster_hint,
            "detected hardware profile"
        );
        profile
    }
}

fn recommend_backend(profile: &HardwareProfile) -> HardwareBackendType {
    if profile.cluster_hint {
        HardwareBackendType::Distributed
    } else if profile.has_gpu {
        HardwareBackendType::Gpu
    } else if profile.cpu_cores >= 8 && profile.total_memory_gb >= 8.0 {
        HardwareBackendType::CpuRamOptimized
    } else if profile.cpu_cores >= 4 && profile.total_memory_gb >= 2.0 {
        HardwareBackendType::MultiThreadedCpu
    } else {
        HardwareBackendType::Cpu
    }
}

fn read_env_bool(key: &str) -> bool {
    std::env::var(key)
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn log_backend_selection(kind: &HardwareBackendType, profile: &HardwareProfile) {
    info!(
        target: "w1z4rdv1510n::hardware",
        backend = ?kind,
        cpu_cores = profile.cpu_cores,
        memory_gb = profile.total_memory_gb,
        has_gpu = profile.has_gpu,
        cluster_hint = profile.cluster_hint,
        "hardware backend resolved"
    );
    match kind {
        HardwareBackendType::Gpu => {
            if !profile.has_gpu {
                warn!(
                    target: "w1z4rdv1510n::hardware",
                    "GPU backend selected but no GPU detected; ensure SIMFUTURES_HAS_GPU=1 if this is intentional"
                );
            }
        }
        HardwareBackendType::Distributed => {
            if !profile.cluster_hint {
                warn!(
                    target: "w1z4rdv1510n::hardware",
                    "Distributed backend selected without cluster hints; set SIMFUTURES_DISTRIBUTED=1 or run under a scheduler"
                );
            }
        }
        _ => {}
    }
}

fn gpu_chunk_size(cpu_cores: usize, memory_gb: f64) -> usize {
    let base = 4096;
    let mem_factor = (memory_gb / 8.0).clamp(0.5, 4.0);
    (base as f64 * mem_factor * (cpu_cores as f64).sqrt()).round() as usize
}

fn distributed_chunk_size(cpu_cores: usize, memory_gb: f64) -> usize {
    let base = 1024;
    let mem_factor = (memory_gb / 4.0).clamp(0.25, 8.0);
    (base as f64 * mem_factor * (cpu_cores as f64).sqrt()).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(cpu: usize, mem: f64, has_gpu: bool, cluster_hint: bool) -> HardwareProfile {
        HardwareProfile {
            cpu_cores: cpu,
            total_memory_gb: mem,
            has_gpu,
            cluster_hint,
        }
    }

    #[test]
    fn auto_prefers_gpu_when_available() {
        let p = profile(8, 16.0, true, false);
        assert!(matches!(recommend_backend(&p), HardwareBackendType::Gpu));
    }

    #[test]
    fn auto_prefers_distributed_on_cluster() {
        let p = profile(2, 1.0, false, true);
        assert!(matches!(
            recommend_backend(&p),
            HardwareBackendType::Distributed
        ));
    }

    #[test]
    fn auto_prefers_cpu_ram_optimized_when_no_gpu() {
        let p = profile(12, 32.0, false, false);
        assert!(matches!(
            recommend_backend(&p),
            HardwareBackendType::CpuRamOptimized
        ));
    }

    #[test]
    fn auto_selects_multithreaded_for_midrange_cpu() {
        let p = profile(6, 4.0, false, false);
        assert!(matches!(
            recommend_backend(&p),
            HardwareBackendType::MultiThreadedCpu
        ));
    }

    #[test]
    fn auto_falls_back_to_single_cpu_on_low_specs() {
        let p = profile(2, 0.5, false, false);
        assert!(matches!(recommend_backend(&p), HardwareBackendType::Cpu));
    }

    #[test]
    fn distributed_backend_applies_updates() {
        use crate::schema::{DynamicState, ParticleState, Population, SymbolState, Timestamp};
        fn population() -> Population {
            let mut particles = Vec::new();
            for id in 0..4 {
                particles.push(ParticleState {
                    id,
                    current_state: DynamicState {
                        timestamp: Timestamp { unix: 0 },
                        symbol_states: std::collections::HashMap::from([(
                            format!("p{id}"),
                            SymbolState::default(),
                        )]),
                    },
                    energy: 0.0,
                    weight: 1.0,
                    history: None,
                    metadata: Default::default(),
                });
            }
            Population {
                particles,
                temperature: 0.0,
                iteration: 0,
            }
        }

        let backend = DistributedBackend::new(7, 1, 2);
        let mut pop = population();
        backend.map_particles(&mut pop, &|particle| {
            particle.energy += 1.0;
        });
        assert!(pop.particles.iter().all(|p| (p.energy - 1.0).abs() < 1e-6));

        backend.map_particles(&mut pop, &|particle| {
            particle.energy *= 2.0;
        });
        assert!(pop.particles.iter().all(|p| (p.energy - 2.0).abs() < 1e-6));
    }
}
