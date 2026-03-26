use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

#[derive(Debug)]
pub enum CircuitState {
    Closed,
    Degraded,
    Open,
}

const COOLDOWN: Duration = Duration::from_secs(60);
const CHECK_TTL: Duration = Duration::from_secs(5);
const CMD_TIMEOUT: Duration = Duration::from_secs(3);

pub struct ObsidianCli {
    pub vault_name: String,
    pub state: CircuitState,
    failures: u32,
    last_check: Instant,
    last_available: bool,
    open_until: Option<Instant>,
}

impl ObsidianCli {
    pub fn new(vault_name: String) -> Self {
        Self {
            vault_name,
            state: CircuitState::Closed,
            failures: 0,
            last_check: Instant::now() - CHECK_TTL, // force first check
            last_available: false,
            open_until: None,
        }
    }

    /// Record a successful CLI operation. Resets circuit to Closed.
    pub fn record_success(&mut self) {
        self.failures = 0;
        self.state = CircuitState::Closed;
        self.open_until = None;
    }

    /// Record a CLI failure. Transitions Closed→Degraded→Open.
    pub fn record_failure(&mut self) {
        self.failures += 1;
        match self.failures {
            1 => self.state = CircuitState::Degraded,
            _ => {
                self.state = CircuitState::Open;
                self.open_until = Some(Instant::now() + COOLDOWN);
            }
        }
    }

    /// Check if we should delegate operations to Obsidian CLI.
    ///
    /// Returns false when the circuit is open (and cooldown hasn't expired),
    /// or when the Obsidian process isn't running.
    pub fn should_delegate(&mut self) -> bool {
        // If Open, check cooldown
        if matches!(self.state, CircuitState::Open) {
            if let Some(until) = self.open_until {
                if Instant::now() < until {
                    return false;
                }
                // Cooldown expired — transition to Degraded for a retry
                self.state = CircuitState::Degraded;
                self.failures = 1;
                self.open_until = None;
            }
        }

        // Check if Obsidian process is running (cached for CHECK_TTL)
        let running = self.check_process();

        running && !matches!(self.state, CircuitState::Open)
    }

    /// Check whether the Obsidian process is running.
    /// Result is cached for `CHECK_TTL` to avoid spawning pgrep on every call.
    fn check_process(&mut self) -> bool {
        if self.last_check.elapsed() < CHECK_TTL {
            return self.last_available;
        }

        let available = Command::new("pgrep")
            .arg("-x")
            .arg("Obsidian")
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        self.last_check = Instant::now();
        self.last_available = available;
        available
    }

    /// Set a property on a vault note via Obsidian CLI.
    pub async fn property_set(
        &mut self,
        file: &str,
        name: &str,
        value: &str,
    ) -> Result<String> {
        self.run_cli(&["property:set", &format!("name={name}"), &format!("value={value}"), &format!("file={file}")]).await
    }

    /// Append content to today's daily note via Obsidian CLI.
    pub async fn daily_append(&mut self, content: &str) -> Result<String> {
        self.run_cli(&["daily:append", &format!("content={content}")]).await
    }

    /// Execute an Obsidian CLI command with a 3-second timeout.
    async fn run_cli(&mut self, args: &[&str]) -> Result<String> {
        let vault_arg = format!("vault={}", self.vault_name);
        let mut cmd = tokio::process::Command::new("obsidian");
        cmd.arg(&vault_arg);
        for arg in args {
            cmd.arg(arg);
        }

        let result = tokio::time::timeout(CMD_TIMEOUT, cmd.output()).await;

        match result {
            Ok(Ok(output)) if output.status.success() => {
                self.record_success();
                Ok(String::from_utf8_lossy(&output.stdout).into_owned())
            }
            Ok(Ok(output)) => {
                self.record_failure();
                let stderr = String::from_utf8_lossy(&output.stderr);
                bail!("obsidian CLI failed (exit {}): {stderr}", output.status)
            }
            Ok(Err(e)) => {
                self.record_failure();
                bail!("obsidian CLI spawn error: {e}")
            }
            Err(_) => {
                self.record_failure();
                bail!("obsidian CLI timed out after {CMD_TIMEOUT:?}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_starts_closed() {
        let cli = ObsidianCli::new("TestVault".into());
        assert!(matches!(cli.state, CircuitState::Closed));
    }

    #[test]
    fn test_single_failure_degrades() {
        let mut cli = ObsidianCli::new("TestVault".into());
        cli.record_failure();
        assert!(matches!(cli.state, CircuitState::Degraded));
    }

    #[test]
    fn test_two_failures_opens() {
        let mut cli = ObsidianCli::new("TestVault".into());
        cli.record_failure();
        cli.record_failure();
        assert!(matches!(cli.state, CircuitState::Open));
    }

    #[test]
    fn test_success_resets_to_closed() {
        let mut cli = ObsidianCli::new("TestVault".into());
        cli.record_failure();
        assert!(matches!(cli.state, CircuitState::Degraded));
        cli.record_success();
        assert!(matches!(cli.state, CircuitState::Closed));
    }

    #[test]
    fn test_is_available_when_open_returns_false() {
        let mut cli = ObsidianCli::new("TestVault".into());
        cli.record_failure();
        cli.record_failure();
        // Open state — should not be available regardless of process
        assert!(!cli.should_delegate());
    }
}
