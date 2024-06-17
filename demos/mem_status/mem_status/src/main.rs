use tokio::runtime::Builder;
use tokio::time::{interval, Duration};
use std::fs::File;
use std::io::{self, Write, BufRead, BufReader};
use std::path::Path;
use clap::{App, Arg};

async fn write_meminfo_to_log(log_path: &str, interval_seconds: u64) -> io::Result<()> {
    let mut interval = interval(Duration::from_secs(interval_seconds));
    loop {
        interval.tick().await;

        let meminfo = read_meminfo("/proc/meminfo")?;
        write_to_log(log_path, &meminfo)?;
    }
}

fn read_meminfo<P>(filename: P) -> io::Result<String>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let buf = BufReader::new(file);
    let mut contents = String::new();
    for line in buf.lines() {
        contents.push_str(&line?);
        contents.push('\n');
    }
    Ok(contents)
}

fn write_to_log<P>(filename: P, content: &str) -> io::Result<()>
where
    P: AsRef<Path>,
{
    let mut file = File::options().append(true).create(true).open(filename)?;
    file.write_all(content.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

async fn run(log_path: String, interval_seconds: u64) {
    // Start meminfo task
    if let Err(e) = write_meminfo_to_log(&log_path, interval_seconds).await {
        eprintln!("Error: {}", e);
    }
}

fn main() {
    let matches = App::new("Memory Info Logger")
        .version("1.0")
        .about("Logs /proc/meminfo to a specified file at a specified interval")
        .arg(Arg::new("log_file")
            .short('l')
            .long("log")
            .help("Sets the log file")
            .takes_value(true)
            .default_value("./meminfo.log"))
        .arg(Arg::new("interval_seconds")
            .short('i')
            .long("interval")
            .help("Sets the interval in seconds")
            .takes_value(true)
            .default_value("5"))
        .get_matches();

    let log_path = matches.value_of("log_file").unwrap().to_string();
    let interval_seconds = matches.value_of_t("interval_seconds").unwrap_or_else(|e| e.exit());

    // Create tokio runtime
    let runtime = Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create runtime");

    runtime.block_on(run(log_path, interval_seconds));
}
