//! Single-file, random-access Wizard brain container.
//!
//! A `.wbrain` file is not a serialized `Brain`. Neuron bodies are separate
//! records and startup reads only a fixed header plus a compact routing
//! manifest. Two alternating header slots publish manifests crash-safely.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::neuron::{Neuron, NeuronId, PoolId};

const HEADER_BYTES: u64 = 4096;
const MAGIC: &[u8; 8] = b"W1ZBRAIN";
const VERSION: u32 = 1;
const SLOT_BYTES: u64 = 64;
const SLOT_A: u64 = 16;
const SLOT_B: u64 = SLOT_A + SLOT_BYTES;
const NEURON_RECORD: &[u8; 8] = b"W1ZNEUR1";
const MANIFEST_RECORD: &[u8; 8] = b"W1ZMANI1";
const AUXILIARY_RECORD: &[u8; 8] = b"W1ZAUX01";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuxiliaryRecordRef {
    pub offset: u64,
    pub len: u64,
}

/// Compact state allowed to remain resident while neuron bodies sleep.
/// Opaque metadata lets the brain layer persist topology and pool routing
/// structures without coupling the container to those implementations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainContainerManifest {
    pub generation: u64,
    pub tick: u64,
    pub brain_metadata: Vec<u8>,
    pub pools: Vec<PoolContainerManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoolContainerManifest {
    pub pool_id: PoolId,
    pub neuron_count: u32,
    pub neuron_offsets: Vec<Option<u64>>,
    pub labels: Vec<(String, NeuronId)>,
    pub pool_metadata: Vec<u8>,
}

#[derive(Clone, Copy)]
struct HeaderSlot {
    generation: u64,
    offset: u64,
    len: u64,
    digest: [u8; 32],
}

pub struct BrainContainer {
    path: PathBuf,
    file: File,
    manifest: Option<BrainContainerManifest>,
}

impl BrainContainer {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;
        if file.metadata()?.len() < HEADER_BYTES {
            Self::initialize(&mut file)?;
        }
        Self::validate_header(&mut file)?;
        let a = Self::read_slot(&mut file, SLOT_A)?;
        let b = Self::read_slot(&mut file, SLOT_B)?;
        let manifest = [a, b]
            .into_iter()
            .flatten()
            .filter_map(|slot| Self::load_manifest_slot(&mut file, slot).ok())
            .max_by_key(|manifest| manifest.generation);
        Ok(Self {
            path,
            file,
            manifest,
        })
    }

    fn initialize(file: &mut File) -> io::Result<()> {
        file.set_len(HEADER_BYTES)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        file.sync_all()
    }

    fn validate_header(file: &mut File) -> io::Result<()> {
        file.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; 8];
        let mut version = [0u8; 4];
        file.read_exact(&mut magic)?;
        file.read_exact(&mut version)?;
        if &magic != MAGIC || u32::from_le_bytes(version) != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a supported Wizard brain container",
            ));
        }
        Ok(())
    }

    fn read_slot(file: &mut File, at: u64) -> io::Result<Option<HeaderSlot>> {
        file.seek(SeekFrom::Start(at))?;
        let mut raw = [0u8; SLOT_BYTES as usize];
        file.read_exact(&mut raw)?;
        let generation = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let offset = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        let len = u64::from_le_bytes(raw[16..24].try_into().unwrap());
        if generation == 0 || offset < HEADER_BYTES || len == 0 {
            return Ok(None);
        }
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&raw[24..56]);
        Ok(Some(HeaderSlot {
            generation,
            offset,
            len,
            digest,
        }))
    }

    fn load_manifest_slot(file: &mut File, slot: HeaderSlot) -> io::Result<BrainContainerManifest> {
        file.seek(SeekFrom::Start(slot.offset))?;
        let mut marker = [0u8; 8];
        let mut len_raw = [0u8; 8];
        file.read_exact(&mut marker)?;
        file.read_exact(&mut len_raw)?;
        let len = u64::from_le_bytes(len_raw);
        if &marker != MANIFEST_RECORD || len != slot.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest header mismatch",
            ));
        }
        let mut body = vec![
            0u8;
            usize::try_from(len).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "manifest too large")
            })?
        ];
        file.read_exact(&mut body)?;
        if *blake3::hash(&body).as_bytes() != slot.digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest digest mismatch",
            ));
        }
        let manifest: BrainContainerManifest = bincode::deserialize(&body)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if manifest.generation != slot.generation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest generation mismatch",
            ));
        }
        Ok(manifest)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn manifest(&self) -> Option<&BrainContainerManifest> {
        self.manifest.as_ref()
    }

    pub fn append_neuron(&mut self, pool: PoolId, neuron: &Neuron) -> io::Result<u64> {
        let body = bincode::serialize(neuron)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let offset = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(NEURON_RECORD)?;
        self.file.write_all(&pool.to_le_bytes())?;
        self.file.write_all(&neuron.id.to_le_bytes())?;
        self.file.write_all(&(body.len() as u64).to_le_bytes())?;
        self.file.write_all(&body)?;
        Ok(offset)
    }

    pub fn read_neuron_at(&mut self, offset: u64) -> io::Result<(PoolId, Neuron)> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut marker = [0u8; 8];
        let mut pool_raw = [0u8; 4];
        let mut id_raw = [0u8; 4];
        let mut len_raw = [0u8; 8];
        self.file.read_exact(&mut marker)?;
        self.file.read_exact(&mut pool_raw)?;
        self.file.read_exact(&mut id_raw)?;
        self.file.read_exact(&mut len_raw)?;
        if &marker != NEURON_RECORD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "neuron marker mismatch",
            ));
        }
        let pool = u32::from_le_bytes(pool_raw);
        let expected_id = u32::from_le_bytes(id_raw);
        let len = usize::try_from(u64::from_le_bytes(len_raw))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "neuron too large"))?;
        let mut body = vec![0u8; len];
        self.file.read_exact(&mut body)?;
        let mut neuron: Neuron = bincode::deserialize(&body)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if neuron.id != expected_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "neuron id mismatch",
            ));
        }
        neuron.rebuild_terminal_idx();
        Ok((pool, neuron))
    }

    pub fn append_auxiliary<F>(
        &mut self,
        pool: PoolId,
        kind: u32,
        write_body: F,
    ) -> io::Result<AuxiliaryRecordRef>
    where
        F: FnOnce(&mut File) -> io::Result<()>,
    {
        let offset = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(AUXILIARY_RECORD)?;
        self.file.write_all(&pool.to_le_bytes())?;
        self.file.write_all(&kind.to_le_bytes())?;
        self.file.write_all(&0_u64.to_le_bytes())?;
        let body_offset = self.file.stream_position()?;
        write_body(&mut self.file)?;
        let end = self.file.stream_position()?;
        let len = end.checked_sub(body_offset).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "auxiliary record moved backwards",
            )
        })?;
        self.file.seek(SeekFrom::Start(offset + 16))?;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(end))?;
        Ok(AuxiliaryRecordRef { offset, len })
    }

    pub fn read_auxiliary(&mut self, reference: AuxiliaryRecordRef) -> io::Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(reference.offset))?;
        let mut marker = [0_u8; 8];
        let mut pool = [0_u8; 4];
        let mut kind = [0_u8; 4];
        let mut len = [0_u8; 8];
        self.file.read_exact(&mut marker)?;
        self.file.read_exact(&mut pool)?;
        self.file.read_exact(&mut kind)?;
        self.file.read_exact(&mut len)?;
        if &marker != AUXILIARY_RECORD || u64::from_le_bytes(len) != reference.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "auxiliary record header mismatch",
            ));
        }
        let mut body = vec![
            0_u8;
            usize::try_from(reference.len).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "auxiliary record too large")
            })?
        ];
        self.file.read_exact(&mut body)?;
        Ok(body)
    }

    /// Make a manifest current only after its complete body is durable.
    pub fn commit_manifest(&mut self, manifest: BrainContainerManifest) -> io::Result<()> {
        let body = bincode::serialize(&manifest)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let digest = *blake3::hash(&body).as_bytes();
        let offset = self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(MANIFEST_RECORD)?;
        self.file.write_all(&(body.len() as u64).to_le_bytes())?;
        self.file.write_all(&body)?;
        self.file.sync_data()?;

        let slot_at = if manifest.generation % 2 == 0 {
            SLOT_A
        } else {
            SLOT_B
        };
        self.file.seek(SeekFrom::Start(slot_at))?;
        self.file.write_all(&manifest.generation.to_le_bytes())?;
        self.file.write_all(&offset.to_le_bytes())?;
        self.file.write_all(&(body.len() as u64).to_le_bytes())?;
        self.file.write_all(&digest)?;
        self.file.write_all(&[0u8; 8])?;
        self.file.sync_data()?;
        self.manifest = Some(manifest);
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::NeuronKind;

    fn tmpfile(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "w1z4rd_{name}_{}_{}.wbrain",
            std::process::id(),
            nonce
        ))
    }

    fn manifest(generation: u64, offset: u64) -> BrainContainerManifest {
        BrainContainerManifest {
            generation,
            tick: 42,
            brain_metadata: b"topology".to_vec(),
            pools: vec![PoolContainerManifest {
                pool_id: 1,
                neuron_count: 1,
                neuron_offsets: vec![Some(offset)],
                labels: vec![("t:YQ".into(), 0)],
                pool_metadata: b"pool".to_vec(),
            }],
        }
    }

    #[test]
    fn startup_reads_manifest_then_neuron_on_demand() {
        let path = tmpfile("lazy");
        {
            let mut container = BrainContainer::open(&path).unwrap();
            let neuron = Neuron::new_atom(0, "t:YQ".into(), NeuronKind::Excitatory, 42);
            let offset = container.append_neuron(1, &neuron).unwrap();
            container.commit_manifest(manifest(1, offset)).unwrap();
        }
        let mut reopened = BrainContainer::open(&path).unwrap();
        let routing = reopened.manifest().unwrap();
        assert_eq!(routing.pools[0].labels[0], ("t:YQ".into(), 0));
        let offset = routing.pools[0].neuron_offsets[0].unwrap();
        let (pool, neuron) = reopened.read_neuron_at(offset).unwrap();
        assert_eq!(pool, 1);
        assert_eq!(neuron.label, "t:YQ");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn alternating_slots_select_latest_valid_generation() {
        let path = tmpfile("generations");
        {
            let mut container = BrainContainer::open(&path).unwrap();
            let neuron = Neuron::new_atom(0, "first".into(), NeuronKind::Excitatory, 1);
            let offset = container.append_neuron(2, &neuron).unwrap();
            container.commit_manifest(manifest(1, offset)).unwrap();
            let mut newer = manifest(2, offset);
            newer.tick = 99;
            container.commit_manifest(newer).unwrap();
        }
        let reopened = BrainContainer::open(&path).unwrap();
        assert_eq!(reopened.manifest().unwrap().generation, 2);
        assert_eq!(reopened.manifest().unwrap().tick, 99);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn auxiliary_record_streams_without_buffering_its_body() {
        let path = tmpfile("auxiliary");
        let reference;
        {
            let mut container = BrainContainer::open(&path).unwrap();
            reference = container
                .append_auxiliary(7, 11, |writer| writer.write_all(b"cold-routing-ledger"))
                .unwrap();
            assert_eq!(reference.len, 19);
        }
        let mut reopened = BrainContainer::open(&path).unwrap();
        assert_eq!(
            reopened.read_auxiliary(reference).unwrap(),
            b"cold-routing-ledger"
        );
        std::fs::remove_file(path).ok();
    }
}
