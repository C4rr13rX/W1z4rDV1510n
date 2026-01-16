use crate::orchestrator::RunOutcome;
use crate::streaming::ingest::StreamIngestor;
use crate::streaming::processor::StreamingInference;
use std::time::Duration;

pub struct StreamingServiceConfig {
    pub max_batch: usize,
    pub idle_sleep_ms: u64,
    pub exit_on_drain: bool,
}

impl Default for StreamingServiceConfig {
    fn default() -> Self {
        Self {
            max_batch: 32,
            idle_sleep_ms: 50,
            exit_on_drain: false,
        }
    }
}

pub struct StreamingService<I: StreamIngestor> {
    inference: StreamingInference,
    ingestor: I,
    config: StreamingServiceConfig,
}

impl<I: StreamIngestor> StreamingService<I> {
    pub fn new(inference: StreamingInference, ingestor: I, config: StreamingServiceConfig) -> Self {
        Self {
            inference,
            ingestor,
            config,
        }
    }

    pub fn run<F>(&mut self, mut on_outcome: F) -> anyhow::Result<()>
    where
        F: FnMut(RunOutcome),
    {
        loop {
            let batch = self.ingestor.poll_batch(self.config.max_batch)?;
            if batch.items.is_empty() {
                if self.config.exit_on_drain && self.ingestor.is_drained() {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(self.config.idle_sleep_ms));
                continue;
            }
            for envelope in batch.items {
                if let Some(outcome) = self.inference.handle_envelope(envelope)? {
                    on_outcome(outcome);
                }
            }
        }
    }
}
