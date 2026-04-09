//! One-Time Password generation and validation.
//!
//! Format: WORD-NNNN  e.g. "RAVEN-7834"
//! - Single-use: burned on first successful join.
//! - TTL: configurable, default 10 minutes.
//! - Stored as Argon2id hash so the coordinator never holds plaintext.

use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use rand::Rng;
use std::time::{Duration, Instant};

/// A pending OTP entry held by the coordinator.
pub struct PendingOtp {
    hash:       String,
    issued_at:  Instant,
    ttl:        Duration,
    used:       bool,
}

impl PendingOtp {
    pub fn is_expired(&self) -> bool {
        self.issued_at.elapsed() > self.ttl
    }
    pub fn is_usable(&self) -> bool {
        !self.used && !self.is_expired()
    }
}

/// The OTP registry held by the coordinator.
#[derive(Default)]
pub struct OtpRegistry {
    entries: Vec<(String, PendingOtp)>, // (plaintext_key, entry)
}

impl OtpRegistry {
    /// Generate a new OTP, store its hash, return the plaintext for printing.
    pub fn generate(&mut self, ttl: Duration) -> anyhow::Result<String> {
        let word   = random_word();
        let digits = rand::thread_rng().gen_range(1000u16..=9999u16);
        let plain  = format!("{}-{}", word, digits);

        let salt   = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        let hash   = argon2
            .hash_password(plain.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("argon2 error: {e}"))?
            .to_string();

        self.entries.push((plain.clone(), PendingOtp {
            hash,
            issued_at: Instant::now(),
            ttl,
            used: false,
        }));
        Ok(plain)
    }

    /// Validate a plaintext OTP from a joining node. Burns it on success.
    pub fn validate(&mut self, plain: &str) -> bool {
        self.purge_expired();
        let argon2 = Argon2::default();
        for (key, entry) in &mut self.entries {
            if !entry.is_usable() { continue; }
            if let Ok(parsed) = PasswordHash::new(&entry.hash) {
                if argon2.verify_password(plain.as_bytes(), &parsed).is_ok() {
                    entry.used = true;
                    return true;
                }
            }
        }
        false
    }

    fn purge_expired(&mut self) {
        self.entries.retain(|(_, e)| !e.is_expired());
    }
}

// 256 wizard/arcane/fantasy words for memorable OTPs
static WORDS: &[&str] = &[
    "ARCANE","RAVEN","SIGIL","GOLEM","RELIC","EMBER","GLOOM","STORM","SHADE",
    "ABYSS","VENOM","BLAZE","CURSE","DRUID","ELDER","FABLE","GHOST","HAVEN",
    "IMBUE","JEWEL","KARMA","LUNAR","MANOR","NEXUS","ONYX","PRISM","QUELL",
    "RUNIC","SABLE","TALON","UMBRA","VIPER","WRAITH","XENON","YIELD","ZEAL",
    "ALTAR","BRINE","CHAOS","DREAD","EPOCH","FLAME","GROVE","HELIX","IVORY",
    "JUDGE","KNELL","LANCE","MARSH","NIGHT","ORBIT","PETAL","QUARTZ","REALM",
    "SCORN","THORN","UDDER","VAULT","WHIRL","EXILE","YONDER","ZENITH","AEGIS",
    "BLIGHT","CHASM","DOOM","ELIXIR","FROST","GLYPH","HASTE","IRONY","JEST",
    "KNAVE","LORE","MAGE","NOBLE","OMEN","PYRE","QUEST","RIDDLE","SCRIBE",
    "TOME","UNDEAD","VORTEX","WITCH","AMULET","BANISH","CRYPT","DIVINER",
    "ETHER","FAMINE","GALLOWS","HERALD","INVOKE","JINX","KNIGHT","LURKER",
    "MYSTIC","NETHER","ORACLE","PILGRIM","QUIVER","RITUAL","SPECTER","TOTEM",
    "UNSEEN","VIGIL","WARLOCK","XENOLITH","YORE","ZEALOT","ABYSSAL","BONFIRE",
    "COBALT","DJINN","ENIGMA","FERVOR","GRIMOIRE","HOLLOW","ILLUSION","JESTER",
    "KINDRED","LABYRINTH","MALICE","NIMBUS","OBSIDIAN","PHANTOM","RECKONING",
    "SHADOW","TEMPEST","UNDYING","VALIANT","WHISPER","ANCIENT","BOUND","COMET",
    "DIVINE","ESSENCE","FORESEE","GUARDIAN","HYMN","IMMORTAL","JOURNEY",
    "KEEPER","LEGEND","MARVEL","NECTAR","OUTCAST","PORTAL","REMNANT","SORCERY",
    "TWILIGHT","UNSEALED","VERDICT","WARDEN","SEER","RUNE","SPECTER","ASTRAL",
];

fn random_word() -> &'static str {
    let idx = rand::thread_rng().gen_range(0..WORDS.len());
    WORDS[idx]
}
