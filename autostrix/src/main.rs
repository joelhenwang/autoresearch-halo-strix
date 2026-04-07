#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Instant;

use chrono::NaiveDateTime;
use clap::{Parser, Subcommand};
use colored::Colorize;
use comfy_table::presets::UTF8_FULL_CONDENSED;
use comfy_table::{Cell, CellAlignment, ContentArrangement, Table};
use rusqlite::{Connection, params};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "autostrix", about = "Experiment monitoring for halo-strix")]
struct Cli {
    /// Path to the SQLite database
    #[arg(long, default_value = "experiments.db")]
    db: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show current + last 5 completed experiments
    Status,
    /// Full history table
    History,
    /// Top N experiments by val_bpb (lowest first)
    Best {
        /// Number of results
        #[arg(default_value_t = 10)]
        n: usize,
    },
    /// Show queued experiments
    Queue,
    /// Show full details for one experiment
    Show {
        /// Experiment ID
        id: i64,
    },
    /// Summary statistics
    Stats,
    /// Print last 30 lines of run.log
    Tail,
    /// Run one full agent pipeline cycle
    RunCycle {
        /// Experiment code name
        #[arg(long)]
        experiment: String,
    },
    /// Run cycles continuously
    RunLoop {
        /// Maximum number of cycles (0 = infinite)
        #[arg(long, default_value_t = 0)]
        max_cycles: usize,
    },
    /// Save an idea, paper, or URL to the ideas bank for the Researcher
    Idea {
        /// The idea text (paper title, URL, concept, etc.)
        text: Vec<String>,
        /// Tag/category (e.g., "paper", "architecture", "optimizer")
        #[arg(long, short)]
        tag: Option<String>,
    },
    /// List all saved ideas
    Ideas,
    /// Queue a human-directed experiment prompt for the Researcher
    Suggest {
        /// Description of the experiment you want tried
        text: Vec<String>,
        /// Priority (higher = tried sooner)
        #[arg(long, short, default_value_t = 10)]
        priority: i64,
    },
    /// List queued experiment suggestions
    Suggestions,
}

// ---------------------------------------------------------------------------
// Row types
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Experiment {
    id: i64,
    started_at: Option<String>,
    finished_at: Option<String>,
    commit_hash: Option<String>,
    config_json: Option<String>,
    val_bpb: Option<f64>,
    peak_memory_mb: Option<f64>,
    training_seconds: Option<f64>,
    mfu_percent: Option<f64>,
    total_tokens_m: Option<f64>,
    num_params_m: Option<f64>,
    num_steps: Option<i64>,
    status: Option<String>,
    description: Option<String>,
    error_message: Option<String>,
}

#[derive(Debug)]
struct QueueEntry {
    id: i64,
    description: Option<String>,
    priority: Option<i64>,
    created_at: Option<String>,
    config_json: Option<String>,
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn fmt_bpb(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.6}"),
        None => "—".into(),
    }
}

fn fmt_memory_gb(mb: Option<f64>) -> String {
    match mb {
        Some(x) => format!("{:.1} GB", x / 1024.0),
        None => "—".into(),
    }
}

fn fmt_duration(secs: Option<f64>) -> String {
    match secs {
        Some(s) => {
            let total = s as u64;
            let m = total / 60;
            let s = total % 60;
            format!("{m}:{s:02}")
        }
        None => "—".into(),
    }
}

fn colorize_status(status: &str) -> String {
    match status {
        "done" | "keep" => status.green().to_string(),
        "running" => status.yellow().to_string(),
        "crash" | "discard" => status.red().to_string(),
        other => other.to_string(),
    }
}

fn fmt_opt<T: std::fmt::Display>(v: &Option<T>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "—".into(),
    }
}

fn fmt_started(s: &Option<String>) -> String {
    match s {
        Some(ts) => {
            // Try to parse and show a shorter form; fall back to raw string.
            NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S")
                .map(|dt| dt.format("%m-%d %H:%M").to_string())
                .unwrap_or_else(|_| ts.clone())
        }
        None => "—".into(),
    }
}

// ---------------------------------------------------------------------------
// Table builder
// ---------------------------------------------------------------------------

fn new_table() -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL_CONDENSED)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table
}

fn experiment_table(rows: &[Experiment]) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("ID").set_alignment(CellAlignment::Right),
        Cell::new("val_bpb").set_alignment(CellAlignment::Right),
        Cell::new("Status"),
        Cell::new("Memory"),
        Cell::new("Time"),
        Cell::new("Started"),
        Cell::new("Description"),
    ]);
    for e in rows {
        let status_str = e.status.as_deref().unwrap_or("—");
        table.add_row(vec![
            Cell::new(e.id).set_alignment(CellAlignment::Right),
            Cell::new(fmt_bpb(e.val_bpb)).set_alignment(CellAlignment::Right),
            Cell::new(colorize_status(status_str)),
            Cell::new(fmt_memory_gb(e.peak_memory_mb)),
            Cell::new(fmt_duration(e.training_seconds)),
            Cell::new(fmt_started(&e.started_at)),
            Cell::new(e.description.as_deref().unwrap_or("—")),
        ]);
    }
    table
}

// ---------------------------------------------------------------------------
// Database helpers
// ---------------------------------------------------------------------------

fn open_db(path: &PathBuf) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    Ok(conn)
}

fn query_experiments(conn: &Connection, sql: &str, limit: usize) -> Vec<Experiment> {
    let mut stmt = conn.prepare(sql).expect("bad SQL");
    stmt.query_map(params![limit as i64], |row| {
        Ok(Experiment {
            id: row.get(0)?,
            started_at: row.get(1)?,
            finished_at: row.get(2)?,
            commit_hash: row.get(3)?,
            config_json: row.get(4)?,
            val_bpb: row.get(5)?,
            peak_memory_mb: row.get(6)?,
            training_seconds: row.get(7)?,
            mfu_percent: row.get(8)?,
            total_tokens_m: row.get(9)?,
            num_params_m: row.get(10)?,
            num_steps: row.get(11)?,
            status: row.get(12)?,
            description: row.get(13)?,
            error_message: row.get(14)?,
        })
    })
    .expect("query failed")
    .filter_map(|r| r.ok())
    .collect()
}

const SELECT_ALL_COLS: &str = "\
    id, started_at, finished_at, commit_hash, config_json, \
    val_bpb, peak_memory_mb, training_seconds, mfu_percent, \
    total_tokens_m, num_params_m, num_steps, status, description, error_message";

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_status(conn: &Connection) {
    // Currently running
    let running = query_experiments(
        conn,
        &format!(
            "SELECT {SELECT_ALL_COLS} FROM experiments \
             WHERE status = 'running' ORDER BY started_at DESC LIMIT ?1"
        ),
        10,
    );

    if running.is_empty() {
        println!("{}", "No experiment currently running.".dimmed());
    } else {
        println!("{}", "Running:".bold().yellow());
        println!("{experiment_table}", experiment_table = experiment_table(&running));
    }

    // Last 5 completed
    let recent = query_experiments(
        conn,
        &format!(
            "SELECT {SELECT_ALL_COLS} FROM experiments \
             WHERE status != 'running' ORDER BY finished_at DESC LIMIT ?1"
        ),
        5,
    );

    if !recent.is_empty() {
        println!("\n{}", "Recent completed:".bold());
        println!("{}", experiment_table(&recent));
    }

    // Summary line
    let best_bpb: Option<f64> = conn
        .query_row(
            "SELECT MIN(val_bpb) FROM experiments WHERE status IN ('done','keep')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(None);
    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM experiments", [], |r| r.get(0))
        .unwrap_or(0);
    let kept: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM experiments WHERE status = 'keep'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);

    let keep_rate = if total > 0 {
        format!("{:.0}%", kept as f64 / total as f64 * 100.0)
    } else {
        "—".into()
    };

    println!(
        "\n{} best_bpb={} total={} keep_rate={}",
        "Summary:".bold(),
        fmt_bpb(best_bpb).green(),
        total,
        keep_rate
    );
}

fn cmd_history(conn: &Connection) {
    let rows = query_experiments(
        conn,
        &format!(
            "SELECT {SELECT_ALL_COLS} FROM experiments ORDER BY id DESC LIMIT ?1"
        ),
        10000,
    );
    if rows.is_empty() {
        println!("No experiments found.");
    } else {
        println!("{}", experiment_table(&rows));
    }
}

fn cmd_best(conn: &Connection, n: usize) {
    let rows = query_experiments(
        conn,
        &format!(
            "SELECT {SELECT_ALL_COLS} FROM experiments \
             WHERE val_bpb IS NOT NULL ORDER BY val_bpb ASC LIMIT ?1"
        ),
        n,
    );
    if rows.is_empty() {
        println!("No experiments with val_bpb recorded.");
    } else {
        println!("{} {} experiments by val_bpb:", "Top".bold(), rows.len());
        println!("{}", experiment_table(&rows));
    }
}

fn cmd_queue(conn: &Connection) {
    let mut stmt = conn
        .prepare(
            "SELECT id, config_json, description, priority, created_at \
             FROM experiment_queue ORDER BY priority DESC, id ASC",
        )
        .expect("bad SQL");

    let entries: Vec<QueueEntry> = stmt
        .query_map([], |row| {
            Ok(QueueEntry {
                id: row.get(0)?,
                config_json: row.get(1)?,
                description: row.get(2)?,
                priority: row.get(3)?,
                created_at: row.get(4)?,
            })
        })
        .expect("query failed")
        .filter_map(|r| r.ok())
        .collect();

    if entries.is_empty() {
        println!("Queue is empty.");
        return;
    }

    let mut table = new_table();
    table.set_header(vec!["ID", "Priority", "Description", "Created"]);
    for e in &entries {
        table.add_row(vec![
            Cell::new(e.id).set_alignment(CellAlignment::Right),
            Cell::new(fmt_opt(&e.priority)).set_alignment(CellAlignment::Right),
            Cell::new(e.description.as_deref().unwrap_or("—")),
            Cell::new(fmt_started(&e.created_at)),
        ]);
    }
    println!("{table}");
}

fn cmd_show(conn: &Connection, id: i64) {
    let e = {
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {SELECT_ALL_COLS} FROM experiments WHERE id = ?1"
            ))
            .expect("bad SQL");
        stmt.query_row(params![id], |row| {
            Ok(Experiment {
                id: row.get(0)?,
                started_at: row.get(1)?,
                finished_at: row.get(2)?,
                commit_hash: row.get(3)?,
                config_json: row.get(4)?,
                val_bpb: row.get(5)?,
                peak_memory_mb: row.get(6)?,
                training_seconds: row.get(7)?,
                mfu_percent: row.get(8)?,
                total_tokens_m: row.get(9)?,
                num_params_m: row.get(10)?,
                num_steps: row.get(11)?,
                status: row.get(12)?,
                description: row.get(13)?,
                error_message: row.get(14)?,
            })
        })
    };

    let e = match e {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Experiment {id} not found.");
            std::process::exit(1);
        }
    };

    let status_str = e.status.as_deref().unwrap_or("—");
    println!("{} {}", "Experiment".bold(), e.id.to_string().bold());
    println!("  Status:       {}", colorize_status(status_str));
    println!("  val_bpb:      {}", fmt_bpb(e.val_bpb));
    println!("  Memory:       {}", fmt_memory_gb(e.peak_memory_mb));
    println!("  Time:         {}", fmt_duration(e.training_seconds));
    println!("  MFU:          {}", fmt_opt(&e.mfu_percent.map(|v| format!("{v:.1}%"))));
    println!("  Params:       {}", fmt_opt(&e.num_params_m.map(|v| format!("{v:.1}M"))));
    println!("  Tokens:       {}", fmt_opt(&e.total_tokens_m.map(|v| format!("{v:.0}M"))));
    println!("  Steps:        {}", fmt_opt(&e.num_steps));
    println!("  Started:      {}", fmt_opt(&e.started_at));
    println!("  Finished:     {}", fmt_opt(&e.finished_at));
    println!("  Commit:       {}", fmt_opt(&e.commit_hash));
    println!("  Description:  {}", fmt_opt(&e.description));

    if let Some(err) = &e.error_message {
        println!("  {}: {}", "Error".red().bold(), err);
    }

    if let Some(cfg) = &e.config_json {
        println!("\n{}:", "Config".bold());
        println!("{cfg}");
    }
}

fn cmd_stats(conn: &Connection) {
    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM experiments", [], |r| r.get(0))
        .unwrap_or(0);
    let completed: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM experiments WHERE status IN ('done','keep','discard')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let crashed: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM experiments WHERE status = 'crash'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let best_bpb: Option<f64> = conn
        .query_row(
            "SELECT MIN(val_bpb) FROM experiments WHERE status IN ('done','keep')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(None);
    let avg_time: Option<f64> = conn
        .query_row(
            "SELECT AVG(training_seconds) FROM experiments WHERE training_seconds IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap_or(None);
    let total_tokens: Option<f64> = conn
        .query_row(
            "SELECT SUM(total_tokens_m) FROM experiments WHERE total_tokens_m IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap_or(None);

    println!("{}", "Experiment Statistics".bold());
    println!("  Total runs:         {total}");
    println!("  Completed:          {completed}");
    println!("  Crashed:            {crashed}");
    println!("  Best val_bpb:       {}", fmt_bpb(best_bpb).green());
    println!("  Avg training time:  {}", fmt_duration(avg_time));
    println!(
        "  Total tokens:       {}",
        fmt_opt(&total_tokens.map(|v| format!("{v:.0}M")))
    );
}

fn cmd_tail() {
    let path = PathBuf::from("run.log");
    if !path.exists() {
        eprintln!("run.log not found in current directory.");
        std::process::exit(1);
    }

    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("Failed to read run.log: {e}");
        std::process::exit(1);
    });

    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(30);
    for line in &lines[start..] {
        println!("{line}");
    }
}

// ---------------------------------------------------------------------------
// Orchestrator: agent roles and permissions
// ---------------------------------------------------------------------------

const AGENTS: &[&str] = &["researcher", "planner", "engineer", "trainer", "reporter", "reviewer"];

fn allowed_tools_for_role(role: &str) -> &'static str {
    match role {
        "researcher" => "Read,Bash(autostrix *),Bash(ls *),Bash(cat *),Glob,Grep,Write",
        "planner"    => "Read,Write,Glob,Grep",
        "engineer"   => "Read,Write,Edit,Bash(uv run *),Bash(python *),Glob,Grep",
        "trainer"    => "Read,Write,Bash(uv run *),Bash(grep *),Bash(tail *),Glob",
        "reporter"   => "Read,Write,Bash(autostrix *),Bash(grep *),Bash(tail *),Glob,Grep",
        "reviewer"   => "Read,Write,Edit,Bash(git *),Glob,Grep",
        _            => "Read,Glob,Grep",
    }
}

fn max_turns_for_role(role: &str) -> &'static str {
    match role {
        "researcher" => "20",
        "planner"    => "15",
        "engineer"   => "40",
        "trainer"    => "30",
        "reporter"   => "15",
        "reviewer"   => "20",
        _            => "10",
    }
}

fn agent_prompt(role: &str, exp_dir: &str) -> String {
    match role {
        "researcher" => format!(
            "You are working on experiment folder: {exp_dir}\n\n\
             1. Run `autostrix best 10` and `autostrix history` to study past experiments.\n\
             2. Randomly pick 2-3 experiments that are NOT in the top 5 and read their HYPOTHESIS.md and RESULTS.md.\n\
             3. Run `autostrix ideas` to check if the human left any ideas, papers, or URLs for you to explore.\n\
             4. Run `autostrix suggestions` to check if the human queued a specific experiment direction. \
                If there are pending suggestions, strongly consider implementing the highest-priority one.\n\
             5. Read `src/components/` files to know what building blocks are available.\n\
             6. Write {exp_dir}/HYPOTHESIS.md following the schema at agents/schemas/HYPOTHESIS.md.\n\
             7. Write {exp_dir}/REVIEW_researcher.md reviewing your session."
        ),
        "planner" => format!(
            "You are working on experiment folder: {exp_dir}\n\n\
             1. Read {exp_dir}/HYPOTHESIS.md.\n\
             2. Read configs/baseline.toml for the default configuration.\n\
             3. Write {exp_dir}/PLAN.md following the schema at agents/schemas/PLAN.md.\n\
             4. Write {exp_dir}/config.toml — a complete, runnable TOML config for this experiment.\n\
             5. Self-review: re-read your PLAN.md and config.toml for correctness.\n\
             6. Write {exp_dir}/REVIEW_planner.md reviewing your session."
        ),
        "engineer" => format!(
            "You are working on experiment folder: {exp_dir}\n\n\
             1. Read {exp_dir}/PLAN.md.\n\
             2. Implement the new component(s) in src/components/ using the @register decorator.\n\
             3. Run smoke test: uv run -m src.train --config {exp_dir}/config.toml --smoke\n\
             4. Fix any issues and re-run until smoke_test: PASS.\n\
             5. Write {exp_dir}/README.md explaining the new component.\n\
             6. Write {exp_dir}/REVIEW_engineer.md reviewing your session."
        ),
        "trainer" => format!(
            "You are working on experiment folder: {exp_dir}\n\n\
             1. Read {exp_dir}/PLAN.md and {exp_dir}/config.toml.\n\
             2. Run smoke test: uv run -m src.train --config {exp_dir}/config.toml --smoke 2>&1\n\
             3. If smoke fails, attempt to fix config.toml or code (max 3 attempts).\n\
             4. Write {exp_dir}/SMOKE_RESULTS.md with smoke test outcome.\n\
             5. If smoke passed, run full training:\n\
                uv run -m src.experiment --config {exp_dir}/config.toml > {exp_dir}/run.log 2>&1\n\
             6. Verify: grep \"^val_bpb:\" {exp_dir}/run.log\n\
             7. Write {exp_dir}/REVIEW_trainer.md reviewing your session."
        ),
        "reporter" => format!(
            "You are working on experiment folder: {exp_dir}\n\n\
             1. Read {exp_dir}/run.log and extract metrics with grep.\n\
             2. Run `autostrix best 10` for comparison context.\n\
             3. Read {exp_dir}/HYPOTHESIS.md to compare expected vs actual results.\n\
             4. Write {exp_dir}/RESULTS.md following the schema at agents/schemas/RESULTS.md.\n\
             5. Write {exp_dir}/REVIEW_reporter.md reviewing your session."
        ),
        "reviewer" => format!(
            "You are working on experiment folder: {exp_dir}\n\n\
             1. Read all REVIEW_*.md files in {exp_dir}/.\n\
             2. Read {exp_dir}/RESULTS.md and {exp_dir}/HYPOTHESIS.md.\n\
             3. Evaluate each agent's performance.\n\
             4. If an agent needs instruction updates, edit agents/<role>/CLAUDE.md and git commit.\n\
             5. Write {exp_dir}/REVIEW_reviewer.md with your final cycle summary."
        ),
        _ => format!("Work on experiment: {exp_dir}"),
    }
}

fn run_agent(role: &str, exp_dir: &str, step: usize, total: usize) -> Result<(), String> {
    let prompt = agent_prompt(role, exp_dir);
    let system_prompt_file = format!("agents/{role}/CLAUDE.md");

    if !Path::new(&system_prompt_file).exists() {
        return Err(format!("Agent instructions not found: {system_prompt_file}"));
    }

    print!("  [{step}/{total}] {role}... ", step = step, total = total, role = role);
    let start = Instant::now();

    let mut child = ProcessCommand::new("claude")
        .arg("-p")
        .arg(&prompt)
        .arg("--model")
        .arg("opus")
        .arg("--system-prompt-file")
        .arg(&system_prompt_file)
        .arg("--output-format")
        .arg("text")
        .arg("--permission-mode")
        .arg("acceptEdits")
        .arg("--allowedTools")
        .arg(allowed_tools_for_role(role))
        .arg("--max-turns")
        .arg(max_turns_for_role(role))
        .arg("--no-session-persistence")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn claude for {role}: {e}"))?;

    // Stream stdout for live output
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let mut first_line = true;
        for line in reader.lines() {
            if let Ok(line) = line {
                if first_line {
                    println!(); // newline after "running..."
                    first_line = false;
                }
                println!("        {line}");
            }
        }
    }

    let status = child.wait().map_err(|e| format!("Agent {role} failed: {e}"))?;
    let elapsed = start.elapsed().as_secs();

    if status.success() {
        println!("  [{step}/{total}] {role}... {} ({elapsed}s)",
                 "done".green(), step = step, total = total, role = role);
        Ok(())
    } else {
        let msg = format!("{role} exited with status {status}");
        println!("  [{step}/{total}] {role}... {} ({msg})",
                 "FAILED".red(), step = step, total = total, role = role);
        Err(msg)
    }
}

fn plan_requires_new_code(plan_path: &str) -> bool {
    let content = std::fs::read_to_string(plan_path).unwrap_or_default();
    content.to_lowercase().contains("new-component")
}

fn verify_file_exists(path: &str) -> Result<(), String> {
    if Path::new(path).exists() {
        Ok(())
    } else {
        Err(format!("Expected file not produced: {path}"))
    }
}

fn init_all_tables(conn: &Connection) {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            current_agent TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            agents_completed TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tag TEXT,
            created_at TEXT NOT NULL,
            used_in_cycle INTEGER
        );
        CREATE TABLE IF NOT EXISTS suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            priority INTEGER DEFAULT 10,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            used_in_cycle INTEGER
        );"
    ).expect("Failed to create tables");
}

fn cmd_run_cycle(db_path: &PathBuf, experiment_name: &str) {
    let exp_dir = format!("experiments/{experiment_name}");
    std::fs::create_dir_all(&exp_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create {exp_dir}: {e}");
        std::process::exit(1);
    });

    let conn = open_db(db_path).unwrap_or_else(|e| {
        eprintln!("Failed to open database: {e}");
        std::process::exit(1);
    });
    init_all_tables(&conn);

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO cycles (experiment_name, started_at, status) VALUES (?1, ?2, 'running')",
        params![experiment_name, now],
    ).expect("Failed to insert cycle");
    let cycle_id: i64 = conn.last_insert_rowid();

    println!("{}", format!("[cycle #{cycle_id}: {experiment_name}]").bold());

    let mut completed_agents: Vec<String> = Vec::new();
    let mut failed = false;

    // Determine steps (engineer may be skipped)
    let agent_sequence = vec![
        ("researcher", true),
        ("planner", true),
        ("engineer", false), // conditional
        ("trainer", true),
        ("reporter", true),
        ("reviewer", true),
    ];

    let mut step = 0;
    let total_steps = 6; // max

    for (role, always_run) in &agent_sequence {
        step += 1;

        // Engineer is conditional on PLAN.md
        if *role == "engineer" {
            let plan_path = format!("{exp_dir}/PLAN.md");
            if !plan_requires_new_code(&plan_path) {
                println!("  [{step}/{total_steps}] {role}... {} (config-only)", "skipped".dimmed());
                continue;
            }
        }

        // Skip reporter/reviewer if trainer didn't produce output
        if (*role == "reporter" || *role == "reviewer") && failed {
            println!("  [{step}/{total_steps}] {role}... {} (prior failure)", "skipped".dimmed());
            continue;
        }

        // Update cycle status
        conn.execute(
            "UPDATE cycles SET current_agent = ?1 WHERE id = ?2",
            params![*role, cycle_id],
        ).ok();

        match run_agent(role, &exp_dir, step, total_steps) {
            Ok(()) => {
                completed_agents.push(role.to_string());
            }
            Err(e) => {
                eprintln!("  Agent {role} failed: {e}");
                failed = true;
                // Don't abort the whole cycle for researcher/planner failures
                if *role == "trainer" {
                    // If trainer fails, skip reporter/reviewer
                    continue;
                }
            }
        }

        // Verify expected outputs
        let required_files: Vec<String> = match *role {
            "researcher" => vec![format!("{exp_dir}/HYPOTHESIS.md")],
            "planner" => vec![format!("{exp_dir}/PLAN.md"), format!("{exp_dir}/config.toml")],
            "trainer" => vec![format!("{exp_dir}/SMOKE_RESULTS.md")],
            "reporter" => vec![format!("{exp_dir}/RESULTS.md")],
            _ => vec![],
        };

        for f in &required_files {
            if let Err(e) = verify_file_exists(f) {
                eprintln!("  Warning: {e}");
            }
        }
    }

    // Update cycle status
    let final_status = if failed { "failed" } else { "completed" };
    let now = chrono::Utc::now().to_rfc3339();
    let agents_json = serde_json::to_string(&completed_agents).unwrap_or_default();
    conn.execute(
        "UPDATE cycles SET finished_at = ?1, status = ?2, agents_completed = ?3, current_agent = NULL WHERE id = ?4",
        params![now, final_status, agents_json, cycle_id],
    ).ok();

    println!("\n{}", format!("[cycle #{cycle_id}: {final_status}]").bold());
}

fn cmd_run_loop(db_path: &PathBuf, max_cycles: usize) {
    let mut cycle = 0;
    loop {
        cycle += 1;
        if max_cycles > 0 && cycle > max_cycles {
            println!("Reached max cycles ({max_cycles}). Stopping.");
            break;
        }

        // Generate experiment name from cycle number and timestamp
        let ts = chrono::Local::now().format("%m%d-%H%M");
        let experiment_name = format!("cycle-{cycle:03}-{ts}");

        println!("\n{}", "=".repeat(60));
        cmd_run_cycle(db_path, &experiment_name);
        println!("{}", "=".repeat(60));
    }
}

// ---------------------------------------------------------------------------
// Ideas bank & suggestions
// ---------------------------------------------------------------------------

fn cmd_idea(db_path: &PathBuf, text: &[String], tag: Option<&str>) {
    let conn = open_db(db_path).expect("Failed to open database");
    init_all_tables(&conn);
    let text = text.join(" ");
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO ideas (text, tag, created_at) VALUES (?1, ?2, ?3)",
        params![text, tag, now],
    ).expect("Failed to insert idea");
    let id: i64 = conn.last_insert_rowid();
    let tag_str = tag.map(|t| format!(" [{}]", t.cyan())).unwrap_or_default();
    println!("{} Idea #{id} saved{tag_str}", "OK".green().bold());
    println!("   {text}");
}

fn cmd_ideas(db_path: &PathBuf) {
    let conn = open_db(db_path).expect("Failed to open database");
    init_all_tables(&conn);
    let mut stmt = conn
        .prepare("SELECT id, text, tag, created_at, used_in_cycle FROM ideas ORDER BY id DESC")
        .expect("bad SQL");
    let rows: Vec<(i64, String, Option<String>, String, Option<i64>)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
        })
        .expect("query failed")
        .filter_map(|r| r.ok())
        .collect();

    if rows.is_empty() {
        println!("{}", "No ideas saved yet. Use `autostrix idea <text>` to add one.".dimmed());
        return;
    }

    println!("{} ({} total)\n", "Ideas Bank".bold(), rows.len());
    for (id, text, tag, created_at, used) in &rows {
        let tag_str = tag.as_ref().map(|t| format!(" [{}]", t.cyan())).unwrap_or_default();
        let used_str = used.map(|c| format!(" (used in cycle #{c})").dimmed().to_string()).unwrap_or_default();
        let date = fmt_started(&Some(created_at.clone()));
        println!("  #{id}{tag_str} {date}{used_str}");
        // Wrap long text
        let display = if text.len() > 120 { format!("{}...", &text[..117]) } else { text.clone() };
        println!("    {display}");
        println!();
    }
}

fn cmd_suggest(db_path: &PathBuf, text: &[String], priority: i64) {
    let conn = open_db(db_path).expect("Failed to open database");
    init_all_tables(&conn);
    let text = text.join(" ");
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO suggestions (text, priority, created_at) VALUES (?1, ?2, ?3)",
        params![text, priority, now],
    ).expect("Failed to insert suggestion");
    let id: i64 = conn.last_insert_rowid();
    println!("{} Suggestion #{id} queued (priority: {priority})", "OK".green().bold());
    println!("   {text}");
}

fn cmd_suggestions(db_path: &PathBuf) {
    let conn = open_db(db_path).expect("Failed to open database");
    init_all_tables(&conn);
    let mut stmt = conn
        .prepare(
            "SELECT id, text, priority, created_at, status, used_in_cycle \
             FROM suggestions ORDER BY \
             CASE WHEN status='pending' THEN 0 ELSE 1 END, priority DESC, id ASC"
        )
        .expect("bad SQL");
    let rows: Vec<(i64, String, i64, String, String, Option<i64>)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?))
        })
        .expect("query failed")
        .filter_map(|r| r.ok())
        .collect();

    if rows.is_empty() {
        println!("{}", "No suggestions queued. Use `autostrix suggest <text>` to add one.".dimmed());
        return;
    }

    let mut table = new_table();
    table.set_header(vec!["ID", "Pri", "Status", "Description"]);
    for (id, text, priority, _created, status, used) in &rows {
        let status_display = match status.as_str() {
            "pending" => "pending".yellow().to_string(),
            "used" => format!("used (cycle #{})", used.unwrap_or(0)).green().to_string(),
            other => other.to_string(),
        };
        let display = if text.len() > 80 { format!("{}...", &text[..77]) } else { text.clone() };
        table.add_row(vec![
            Cell::new(id).set_alignment(CellAlignment::Right),
            Cell::new(priority).set_alignment(CellAlignment::Right),
            Cell::new(status_display),
            Cell::new(display),
        ]);
    }
    println!("{table}");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Cmd::Tail => {
            cmd_tail();
            return;
        }
        Cmd::RunCycle { experiment } => {
            cmd_run_cycle(&cli.db, experiment);
            return;
        }
        Cmd::RunLoop { max_cycles } => {
            cmd_run_loop(&cli.db, *max_cycles);
            return;
        }
        Cmd::Idea { text, tag } => {
            cmd_idea(&cli.db, text, tag.as_deref());
            return;
        }
        Cmd::Ideas => {
            cmd_ideas(&cli.db);
            return;
        }
        Cmd::Suggest { text, priority } => {
            cmd_suggest(&cli.db, text, *priority);
            return;
        }
        Cmd::Suggestions => {
            cmd_suggestions(&cli.db);
            return;
        }
        _ => {}
    }

    let conn = open_db(&cli.db).unwrap_or_else(|e| {
        eprintln!("Failed to open database {:?}: {e}", cli.db);
        std::process::exit(1);
    });

    match &cli.command {
        Cmd::Status => cmd_status(&conn),
        Cmd::History => cmd_history(&conn),
        Cmd::Best { n } => cmd_best(&conn, *n),
        Cmd::Queue => cmd_queue(&conn),
        Cmd::Show { id } => cmd_show(&conn, *id),
        Cmd::Stats => cmd_stats(&conn),
        _ => unreachable!(),
    }
}
