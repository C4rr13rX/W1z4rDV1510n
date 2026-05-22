//! WAL replay → in-memory brain reconstruction per [`ARCHITECTURE.md`] §17.9.
//!
//! Stage 17.1 implements *event accounting* — we replay every event and
//! return a [`RecoveryStats`] block summarising what was seen.  The actual
//! Brain-rehydration callbacks (apply each event to a partially-built
//! Brain) are intentionally deferred to the Brain-wiring task in the next
//! commit, because they need to mutate `Pool` internals that aren't fully
//! exposed yet.
//!
//! The stats are useful in their own right:
//! - **Crash diagnostics**: how many events were in the log; how far past
//!   the last snapshot marker did we get; which event variants dominate.
//! - **Test gate**: the Stage 1 acceptance test asserts that `replay_into_brain`
//!   sees exactly the events the brain previously appended.
//! - **Forward-compat probe**: the replay loop is the canonical place to
//!   detect WAL-format-version drift.

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;

use super::event::WalEvent;
use super::wal::{MmapWalStore, WalReader};

/// Summary of a successful WAL replay.
#[derive(Debug, Default, Clone)]
pub struct RecoveryStats {
    pub events_total:           u64,
    pub events_by_variant:      HashMap<String, u64>,
    pub last_tick:              u64,
    pub last_snapshot_tick:     Option<u64>,
    pub events_since_snapshot:  u64,
    pub bytes_read:             u64,
}

impl RecoveryStats {
    pub fn observe(&mut self, event: &WalEvent) {
        self.events_total += 1;
        *self.events_by_variant
            .entry(event.variant_name().to_string())
            .or_insert(0) += 1;

        match event {
            WalEvent::SnapshotMarker { tick, .. } => {
                self.last_snapshot_tick = Some(*tick);
                self.events_since_snapshot = 0;
            }
            WalEvent::TickAdvanced { new_tick } => {
                if *new_tick > self.last_tick { self.last_tick = *new_tick; }
                self.events_since_snapshot += 1;
            }
            _ => {
                self.events_since_snapshot += 1;
            }
        }
    }
}

/// Replay the WAL at `data_dir/brain.wal`.  Stage 17.1: returns the event
/// summary.  Stage 17.9: per-event apply path lives on the Brain
/// (`Brain::apply_wal_event`); use [`load_events_after_marker`] to read
/// the slice of events past the most recent `SnapshotMarker` and feed
/// them to the brain.
///
/// Returns `Ok(None)` if no WAL exists yet (fresh brain on first run).
pub fn replay_into_brain<P: AsRef<Path>>(data_dir: P)
    -> io::Result<Option<RecoveryStats>>
{
    let path = data_dir.as_ref().join("brain.wal");
    if !path.exists() { return Ok(None); }

    let f = MmapWalStore::open_replay_only(&data_dir)?;
    let mut stats = RecoveryStats::default();
    let mut counting_reader = CountingReader::new(f);
    {
        let reader = WalReader::new(&mut counting_reader);
        for ev in reader {
            let ev = ev?;
            stats.observe(&ev);
        }
    }
    stats.bytes_read = counting_reader.read;
    Ok(Some(stats))
}

/// Stage 17.9 — read the WAL and return the events that come *after*
/// the most recent `SnapshotMarker`.  Used at startup: the brain.bin
/// snapshot reconstructs state up through the snapshot marker; this
/// function returns the tail of events that have to be replayed to
/// bring the brain forward to its true last-known state.
///
/// If the WAL doesn't exist, returns `Ok(vec![])`.  If no marker has
/// been written yet (a fresh WAL or a process that crashed before its
/// first checkpoint), returns ALL events — because there's no snapshot
/// to rely on.
pub fn load_events_after_marker<P: AsRef<Path>>(data_dir: P)
    -> io::Result<Vec<WalEvent>>
{
    let path = data_dir.as_ref().join("brain.wal");
    if !path.exists() { return Ok(Vec::new()); }
    let f = MmapWalStore::open_replay_only(&data_dir)?;
    let mut all_events: Vec<WalEvent> = Vec::new();
    let mut last_marker_idx: Option<usize> = None;
    for (i, ev) in WalReader::new(f).enumerate() {
        let ev = ev?;
        if matches!(ev, WalEvent::SnapshotMarker { .. }) {
            last_marker_idx = Some(i);
        }
        all_events.push(ev);
    }
    let start = last_marker_idx.map(|i| i + 1).unwrap_or(0);
    Ok(all_events.split_off(start))
}

/// Wraps a `Read` to track total bytes consumed.  Used to populate
/// `RecoveryStats::bytes_read`.
struct CountingReader<R: Read> {
    inner: R,
    read:  u64,
}

impl<R: Read> CountingReader<R> {
    fn new(inner: R) -> Self { Self { inner, read: 0 } }
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n as u64;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{NeuronKind, NeuronRef};
    use crate::store::event::TerminalDelta;
    use crate::store::wal::{MmapWalStore, Store};
    use std::env::temp_dir;
    use std::path::PathBuf;

    fn tmpdir_for(test: &str) -> PathBuf {
        let pid = std::process::id();
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let d = temp_dir().join(format!("w1z4rd_brain_recovery_test_{}_{}_{}", test, pid, nano));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn replay_counts_match_appends() {
        let dir = tmpdir_for("counts");
        let s = MmapWalStore::open(&dir).unwrap();

        s.append(&WalEvent::AtomCreated {
            pool_id: 1, id: 0, label: "t:a".into(),
            kind: NeuronKind::Excitatory, born_tick: 0,
        }).unwrap();
        s.append(&WalEvent::AtomCreated {
            pool_id: 1, id: 1, label: "t:b".into(),
            kind: NeuronKind::Excitatory, born_tick: 0,
        }).unwrap();
        s.append(&WalEvent::TerminalReinforced(TerminalDelta {
            src: NeuronRef::new(1, 0), dst: NeuronRef::new(1, 1),
            weight: 0.5, consolidation: 1, last_fired_tick: 1,
        })).unwrap();
        s.append(&WalEvent::TickAdvanced { new_tick: 1 }).unwrap();
        s.append(&WalEvent::SnapshotMarker { tick: 1, wall_time_ms: 0 }).unwrap();
        s.append(&WalEvent::TickAdvanced { new_tick: 2 }).unwrap();
        s.flush().unwrap();
        drop(s);

        let stats = replay_into_brain(&dir).unwrap().unwrap();
        assert_eq!(stats.events_total, 6);
        assert_eq!(stats.events_by_variant.get("AtomCreated").copied(), Some(2));
        assert_eq!(stats.events_by_variant.get("TerminalReinforced").copied(), Some(1));
        assert_eq!(stats.last_tick, 2);
        assert_eq!(stats.last_snapshot_tick, Some(1));
        assert_eq!(stats.events_since_snapshot, 1);
        assert!(stats.bytes_read > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn replay_missing_log_is_none() {
        let dir = tmpdir_for("missing");
        let stats = replay_into_brain(&dir).unwrap();
        assert!(stats.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
