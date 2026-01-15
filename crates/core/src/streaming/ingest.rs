use super::schema::StreamEnvelope;
use anyhow::Result;

#[derive(Debug, Default)]
pub struct StreamIngestBatch {
    pub items: Vec<StreamEnvelope>,
}

pub trait StreamIngestor: Send + Sync {
    fn poll(&mut self) -> Result<Option<StreamEnvelope>>;

    fn poll_batch(&mut self, max_items: usize) -> Result<StreamIngestBatch> {
        let mut batch = StreamIngestBatch::default();
        for _ in 0..max_items.max(1) {
            match self.poll()? {
                Some(item) => batch.items.push(item),
                None => break,
            }
        }
        Ok(batch)
    }
}
