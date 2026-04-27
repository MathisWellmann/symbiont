use tracing_subscriber::EnvFilter;

/// Initialize `tracing` based logging setup, with ANSI color support and pretty printing.
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stdout) // force same stream as println!
        .with_line_number(true)
        .with_ansi(true)
        .with_ansi_sanitization(false)
        .init();
}
