use anyhow::Context;
use clap::Parser;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use w1z4rdv1510n::config::RunConfig;
use w1z4rdv1510n::streaming::{
    PoseCommandConfig, PoseCommandIngestor, StreamEnvelope, StreamIngestor, StreamingInference,
    StreamingService, StreamingServiceConfig,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Run the streaming inference loop over JSONL envelopes")]
struct Args {
    #[arg(long, default_value = "run_config.json")]
    config: PathBuf,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    pose_source: Option<String>,
    #[arg(long)]
    pose_cmd: Option<String>,
    #[arg(long = "pose-arg")]
    pose_args: Vec<String>,
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    #[arg(long, default_value_t = 50)]
    idle_ms: u64,
    #[arg(long)]
    exit_on_drain: bool,
}

struct JsonLineIngestor {
    reader: Box<dyn BufRead + Send + Sync>,
    eof: bool,
}

impl JsonLineIngestor {
    fn new(reader: Box<dyn BufRead + Send + Sync>) -> Self {
        Self { reader, eof: false }
    }
}

impl StreamIngestor for JsonLineIngestor {
    fn poll(&mut self) -> anyhow::Result<Option<StreamEnvelope>> {
        if self.eof {
            return Ok(None);
        }
        loop {
            let mut line = String::new();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                self.eof = true;
                return Ok(None);
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let envelope = serde_json::from_str(trimmed)
                .with_context(|| "invalid JSON stream envelope")?;
            return Ok(Some(envelope));
        }
    }

    fn is_drained(&self) -> bool {
        self.eof
    }
}

enum IngestorKind {
    Json(JsonLineIngestor),
    Pose(PoseCommandIngestor),
}

impl StreamIngestor for IngestorKind {
    fn poll(&mut self) -> anyhow::Result<Option<StreamEnvelope>> {
        match self {
            IngestorKind::Json(ingestor) => ingestor.poll(),
            IngestorKind::Pose(ingestor) => ingestor.poll(),
        }
    }

    fn is_drained(&self) -> bool {
        match self {
            IngestorKind::Json(ingestor) => ingestor.is_drained(),
            IngestorKind::Pose(ingestor) => ingestor.is_drained(),
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let config_data = std::fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read config {:?}", args.config))?;
    let run_config: RunConfig = serde_json::from_str(&config_data)
        .with_context(|| format!("failed to parse config {:?}", args.config))?;
    if !run_config.streaming.enabled {
        anyhow::bail!("streaming.enabled must be true to run the streaming service");
    }

    let using_pose = args.pose_cmd.is_some() || args.pose_source.is_some();
    if using_pose && (args.pose_cmd.is_none() || args.pose_source.is_none()) {
        anyhow::bail!("pose_cmd and pose_source must be provided together");
    }

    let input_is_file = args
        .input
        .as_deref()
        .map(|path| path != Path::new("-"))
        .unwrap_or(false);
    let exit_on_drain = if using_pose {
        args.exit_on_drain
    } else {
        args.exit_on_drain || input_is_file
    };

    let ingestor = if using_pose {
        let mut command = Vec::new();
        command.push(args.pose_cmd.clone().unwrap());
        command.extend(args.pose_args.clone());
        let pose_config = PoseCommandConfig::new(command, args.pose_source.clone().unwrap());
        IngestorKind::Pose(PoseCommandIngestor::spawn(pose_config)?)
    } else {
        let reader: Box<dyn BufRead + Send + Sync> = match args.input.as_deref() {
            Some(path) if path != Path::new("-") => Box::new(BufReader::new(File::open(path)?)),
            _ => Box::new(BufReader::new(io::stdin())),
        };
        IngestorKind::Json(JsonLineIngestor::new(reader))
    };

    let inference = StreamingInference::new(run_config);
    let service_config = StreamingServiceConfig {
        max_batch: args.max_batch,
        idle_sleep_ms: args.idle_ms,
        exit_on_drain,
    };
    let mut service = StreamingService::new(inference, ingestor, service_config);
    service.run(|outcome| {
        tracing::info!(
            target: "w1z4rdv1510n::streaming",
            best_energy = outcome.results.best_energy,
            symbols = outcome.results.best_state.symbol_states.len(),
            "streaming batch processed"
        );
    })?;
    Ok(())
}
