//! Immutable, request-keyed posting lists stored as auxiliary `.wbrain` records.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::neuron::{NeuronId, PoolId};

use super::{AuxiliaryRecordRef, WbrainFile, WbrainNeuronStore};

const POSTING_INDEX_KIND: u32 = 0x504F_5354; // "POST"
const POSTING_INDEX_MAGIC: &[u8; 8] = b"W1ZPOST1";
const HEADER_BYTES: u64 = 24;
const RECORD_HEADER_BYTES: u64 = 24;
const MAX_BUCKETS: u64 = 4 * 1024 * 1024;

pub(crate) struct PostingIndexBuilder {
    path: PathBuf,
    file: File,
    bucket_count: u64,
    bucket_heads: Vec<u64>,
    entries: u64,
    next_record_offset: u64,
}

impl PostingIndexBuilder {
    /// Resume the newest unfinished index for this pool. Records are append
    /// ordered, so one bounded sequential scan reconstructs the bucket heads
    /// without retaining keys or posting lists. A torn final record is
    /// truncated; the caller replays the last value to complete its logical
    /// transaction idempotently.
    pub(crate) fn resume_latest(
        directory: &Path,
        pool_id: PoolId,
        scope: &str,
    ) -> io::Result<Option<(Self, Option<NeuronId>)>> {
        let prefix = format!(".pool-{pool_id}.posting-index-{scope}-");
        let mut candidates = Vec::new();
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".tmp") {
                candidates.push(entry.path());
            }
        }
        candidates.sort_by_key(|path| {
            std::fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
        });
        let Some(path) = candidates.pop() else {
            return Ok(None);
        };
        Self::resume(path).map(Some)
    }

    fn resume(path: PathBuf) -> io::Result<(Self, Option<NeuronId>)> {
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut header = [0_u8; HEADER_BYTES as usize];
        file.read_exact(&mut header)?;
        if &header[..8] != POSTING_INDEX_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid unfinished posting index magic",
            ));
        }
        let bucket_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
        if bucket_count == 0 || !bucket_count.is_power_of_two() || bucket_count > MAX_BUCKETS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid unfinished posting bucket count",
            ));
        }
        let mut bucket_heads = vec![0_u64; usize::try_from(bucket_count).unwrap()];
        let records_start = HEADER_BYTES + bucket_count * 8;
        let file_len = file.metadata()?.len();
        if file_len < records_start {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unfinished posting index lacks bucket directory",
            ));
        }
        let mut at = records_start;
        let mut entries = 0_u64;
        let mut last_value = None;
        let mut raw = [0_u8; RECORD_HEADER_BYTES as usize];
        while at
            .checked_add(RECORD_HEADER_BYTES)
            .is_some_and(|end| end <= file_len)
        {
            file.seek(SeekFrom::Start(at))?;
            file.read_exact(&mut raw)?;
            let hash = u64::from_le_bytes(raw[8..16].try_into().unwrap());
            let key_len = u32::from_le_bytes(raw[16..20].try_into().unwrap()) as u64;
            let value = u32::from_le_bytes(raw[20..24].try_into().unwrap());
            let Some(end) = at
                .checked_add(RECORD_HEADER_BYTES)
                .and_then(|offset| offset.checked_add(key_len))
            else {
                break;
            };
            if end > file_len {
                break;
            }
            let bucket = usize::try_from(hash & (bucket_count - 1)).unwrap();
            bucket_heads[bucket] = at + 1;
            entries += 1;
            last_value = Some(value);
            at = end;
        }
        if at != file_len {
            file.set_len(at)?;
        }
        file.seek(SeekFrom::Start(at))?;
        Ok((
            Self {
                path,
                file,
                bucket_count,
                bucket_heads,
                entries,
                next_record_offset: at,
            },
            last_value,
        ))
    }

    pub(crate) fn create(
        directory: &Path,
        pool_id: PoolId,
        expected_entries: u64,
    ) -> io::Result<Self> {
        Self::create_inner(directory, pool_id, expected_entries, None)
    }

    pub(crate) fn create_scoped(
        directory: &Path,
        pool_id: PoolId,
        expected_entries: u64,
        scope: &str,
    ) -> io::Result<Self> {
        Self::create_inner(directory, pool_id, expected_entries, Some(scope))
    }

    fn create_inner(
        directory: &Path,
        pool_id: PoolId,
        expected_entries: u64,
        scope: Option<&str>,
    ) -> io::Result<Self> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let scope = scope.map_or_else(String::new, |value| format!("{value}-"));
        let path = directory.join(format!(
            ".pool-{pool_id}.posting-index-{scope}{}-{nonce}.tmp",
            std::process::id()
        ));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let bucket_count = expected_entries
            .saturating_mul(2)
            .max(1)
            .next_power_of_two()
            .min(MAX_BUCKETS);
        file.write_all(POSTING_INDEX_MAGIC)?;
        file.write_all(&0_u64.to_le_bytes())?;
        file.write_all(&bucket_count.to_le_bytes())?;
        let zeroes = [0_u8; 64 * 1024];
        let mut remaining = bucket_count * 8;
        while remaining > 0 {
            let n = usize::try_from(remaining.min(zeroes.len() as u64)).unwrap();
            file.write_all(&zeroes[..n])?;
            remaining -= n as u64;
        }
        let next_record_offset = HEADER_BYTES + bucket_count * 8;
        Ok(Self {
            path,
            file,
            bucket_count,
            bucket_heads: vec![0; usize::try_from(bucket_count).unwrap()],
            entries: 0,
            next_record_offset,
        })
    }

    pub(crate) fn insert(&mut self, key: &[u8], value: NeuronId) -> io::Result<()> {
        let digest = blake3::hash(key);
        let hash = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
        let bucket = hash & (self.bucket_count - 1);
        let bucket_index = usize::try_from(bucket).unwrap();
        let old_head = self.bucket_heads[bucket_index];
        let record = self.next_record_offset;
        self.file.write_all(&old_head.to_le_bytes())?;
        self.file.write_all(&hash.to_le_bytes())?;
        self.file.write_all(
            &u32::try_from(key.len())
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "posting key too large"))?
                .to_le_bytes(),
        )?;
        self.file.write_all(&value.to_le_bytes())?;
        self.file.write_all(key)?;
        self.bucket_heads[bucket_index] = record + 1;
        self.next_record_offset = record
            .checked_add(RECORD_HEADER_BYTES)
            .and_then(|offset| offset.checked_add(key.len() as u64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "posting index overflow"))?;
        self.entries += 1;
        Ok(())
    }

    pub(crate) fn finish(
        mut self,
        destination: &Arc<WbrainFile>,
        pool_id: PoolId,
    ) -> io::Result<AuxiliaryRecordRef> {
        self.file.seek(SeekFrom::Start(8))?;
        self.file.write_all(&self.entries.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(HEADER_BYTES))?;
        for head in &self.bucket_heads {
            self.file.write_all(&head.to_le_bytes())?;
        }
        self.file.flush()?;
        self.file.seek(SeekFrom::Start(0))?;
        let reference = destination.append_auxiliary(pool_id, POSTING_INDEX_KIND, |writer| {
            io::copy(&mut self.file, writer).map(|_| ())
        })?;
        drop(self.file);
        std::fs::remove_file(&self.path).ok();
        Ok(reference)
    }
}

pub(crate) fn lookup(
    store: &WbrainNeuronStore,
    reference: AuxiliaryRecordRef,
    key: &[u8],
    limit: usize,
) -> io::Result<Vec<NeuronId>> {
    let mut header = [0_u8; HEADER_BYTES as usize];
    store.read_auxiliary_exact(reference, 0, &mut header)?;
    if &header[..8] != POSTING_INDEX_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid posting index magic",
        ));
    }
    let entries = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let buckets = u64::from_le_bytes(header[16..24].try_into().unwrap());
    if buckets == 0 || !buckets.is_power_of_two() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid binding posting bucket count",
        ));
    }
    let digest = blake3::hash(key);
    let hash = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
    let bucket = hash & (buckets - 1);
    let mut head = [0_u8; 8];
    store.read_auxiliary_exact(reference, HEADER_BYTES + bucket * 8, &mut head)?;
    let mut next_plus_one = u64::from_le_bytes(head);
    let mut visited = 0_u64;
    let mut found = Vec::new();
    while next_plus_one != 0 && found.len() < limit {
        if visited >= entries {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting index chain cycle",
            ));
        }
        let record = next_plus_one - 1;
        if record
            .checked_add(RECORD_HEADER_BYTES)
            .is_none_or(|end| end > reference.len)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting index record outside auxiliary body",
            ));
        }
        let mut raw = [0_u8; RECORD_HEADER_BYTES as usize];
        store.read_auxiliary_exact(reference, record, &mut raw)?;
        next_plus_one = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let stored_hash = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        let key_len = u32::from_le_bytes(raw[16..20].try_into().unwrap()) as u64;
        let value = u32::from_le_bytes(raw[20..24].try_into().unwrap());
        let key_at = record + RECORD_HEADER_BYTES;
        if key_at
            .checked_add(key_len)
            .is_none_or(|end| end > reference.len)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting index key outside auxiliary body",
            ));
        }
        if stored_hash == hash && key_len == key.len() as u64 {
            let mut stored = vec![0_u8; key.len()];
            store.read_auxiliary_exact(reference, key_at, &mut stored)?;
            if stored == key && found.last().copied() != Some(value) {
                found.push(value);
            }
        }
        visited += 1;
    }
    found.reverse();
    Ok(found)
}

/// Visit every record whose key begins with `prefix` without materializing the
/// index. Records are laid out sequentially after the bucket-head table even
/// though point lookups follow hash chains, so maintenance can scan them with
/// memory bounded by one key.
pub(crate) fn scan_prefix<F>(
    store: &WbrainNeuronStore,
    reference: AuxiliaryRecordRef,
    prefix: &[u8],
    mut visitor: F,
) -> io::Result<()>
where
    F: FnMut(&[u8], NeuronId) -> io::Result<()>,
{
    let mut header = [0_u8; HEADER_BYTES as usize];
    store.read_auxiliary_exact(reference, 0, &mut header)?;
    if &header[..8] != POSTING_INDEX_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid posting index magic",
        ));
    }
    let entries = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let buckets = u64::from_le_bytes(header[16..24].try_into().unwrap());
    if buckets == 0 || !buckets.is_power_of_two() || buckets > MAX_BUCKETS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid posting index bucket count",
        ));
    }
    let mut at = HEADER_BYTES
        .checked_add(buckets.saturating_mul(8))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "posting index overflow"))?;
    for _ in 0..entries {
        if at
            .checked_add(RECORD_HEADER_BYTES)
            .is_none_or(|end| end > reference.len)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting index record outside auxiliary body",
            ));
        }
        let mut raw = [0_u8; RECORD_HEADER_BYTES as usize];
        store.read_auxiliary_exact(reference, at, &mut raw)?;
        let key_len = u32::from_le_bytes(raw[16..20].try_into().unwrap()) as u64;
        let value = u32::from_le_bytes(raw[20..24].try_into().unwrap());
        let key_at = at + RECORD_HEADER_BYTES;
        let end = key_at.checked_add(key_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "posting index key overflow")
        })?;
        if end > reference.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "posting index key outside auxiliary body",
            ));
        }
        let mut key = vec![
            0_u8;
            usize::try_from(key_len).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "posting index key too large")
            })?
        ];
        store.read_auxiliary_exact(reference, key_at, &mut key)?;
        if key.starts_with(prefix) {
            visitor(&key, value)?;
        }
        at = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_directory(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "w1z4rd_posting_{name}_{}_{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn resumes_complete_records_and_truncates_torn_tail() {
        let directory = temp_directory("resume");
        std::fs::create_dir_all(&directory).unwrap();
        let destination = directory.join("brain.wbrain");
        let complete_len;
        let unfinished_path;
        {
            let mut builder =
                PostingIndexBuilder::create_scoped(&directory, 7, 8, "brain.wbrain").unwrap();
            builder.insert(b"alpha", 4).unwrap();
            builder.insert(b"beta", 7).unwrap();
            complete_len = builder.next_record_offset;
            unfinished_path = builder.path.clone();
        }
        {
            let mut file = OpenOptions::new()
                .append(true)
                .open(&unfinished_path)
                .unwrap();
            file.write_all(&[0xAA, 0xBB, 0xCC]).unwrap();
        }

        let (mut resumed, last_value) =
            PostingIndexBuilder::resume_latest(&directory, 7, "brain.wbrain")
            .unwrap()
            .unwrap();
        assert_eq!(last_value, Some(7));
        assert_eq!(resumed.entries, 2);
        assert_eq!(resumed.next_record_offset, complete_len);
        assert_eq!(
            std::fs::metadata(&unfinished_path).unwrap().len(),
            complete_len
        );
        resumed.insert(b"beta", 7).unwrap();
        resumed.insert(b"gamma", 8).unwrap();

        let file = WbrainFile::open(&destination).unwrap();
        let reference = resumed.finish(&file, 7).unwrap();
        let store = file.pool(7);
        assert_eq!(lookup(&store, reference, b"alpha", 8).unwrap(), vec![4]);
        assert_eq!(lookup(&store, reference, b"beta", 8).unwrap(), vec![7]);
        assert_eq!(lookup(&store, reference, b"gamma", 8).unwrap(), vec![8]);
        assert!(!unfinished_path.exists());
        drop(store);
        drop(file);
        std::fs::remove_dir_all(directory).ok();
    }
}
