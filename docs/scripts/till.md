# Wait Till Conditions

The `till` script allows you to wait until specific conditions are met before proceeding. This is useful for scheduling tasks, resource management, and synchronization.

## Usage

```bash
python -m mbapy.scripts.till [sub-command] [options]
```

Or using mbapy-cli:

```bash
mbapy-cli till [sub-command] [options]
```

## Available Sub-commands

### time - Wait for specified duration

Wait for a specified time duration.

**Usage:**
```bash
mbapy-cli till time <time>
```

**Arguments:**
- `time`: Duration to wait (supports multiple formats)

**Supported time formats:**
- Pure numbers: `60` (60 seconds)
- With units: `3s`, `1m2s`, `1h3s`, `1h8m3s`, `2h30m`, `45m`, `1h`

**Examples:**
```bash
# Wait 60 seconds
mbapy-cli till time 60

# Wait 3 seconds
mbapy-cli till time 3s

# Wait 1 minute and 2 seconds
mbapy-cli till time 1m2s

# Wait 1 hour, 8 minutes and 3 seconds
mbapy-cli till time 1h8m3s

# Wait 2 hours and 30 minutes
mbapy-cli till time 2h30m
```

### start-time - Wait until specified start time

Wait until a specific date and time.

**Usage:**
```bash
mbapy-cli till start-time <time>
```

**Arguments:**
- `time`: Start time in format `YYYY-MM-DD HH:MM:SS`

**Examples:**
```bash
# Wait until February 5, 2026 at 14:30:00
mbapy-cli till start-time "2026-02-05 14:30:00"
```

### cuda-mem - Wait until GPU memory is available

Wait until specified amount of GPU memory becomes available.

**Usage:**
```bash
mbapy-cli till cuda-mem <memory> [options]
```

**Arguments:**
- `memory`: Required GPU memory in GB

**Options:**
- `--check-interval`: Check interval in seconds (default: 30)
- `--wait-iter`: Wait iterations after condition met (default: 5)

**Examples:**
```bash
# Wait until 8GB GPU memory is available
mbapy-cli till cuda-mem 8

# Wait for 10GB with custom settings
mbapy-cli till cuda-mem 10 --check-interval 60 --wait-iter 10
```

### cuda - Wait until GPU utilization drops

Wait until GPU utilization drops below specified threshold.

**Usage:**
```bash
mbapy-cli till cuda <max-utilization> [options]
```

**Arguments:**
- `max-utilization`: Maximum GPU utilization percentage

**Options:**
- `--check-interval`: Check interval in seconds (default: 30)
- `--wait-iter`: Wait iterations after condition met (default: 5)

**Examples:**
```bash
# Wait until GPU utilization drops below 10%
mbapy-cli till cuda 10

# Wait with custom settings
mbapy-cli till cuda 20 --check-interval 60 --wait-iter 3
```

### process - Wait until process ends

Wait until a specific process (by PID) ends.

**Usage:**
```bash
mbapy-cli till process <pid> [options]
```

**Arguments:**
- `pid`: Process ID to wait for

**Options:**
- `--check-interval`: Check interval in seconds (default: 10)

**Examples:**
```bash
# Wait for process 12345 to end
mbapy-cli till process 12345

# Check every 5 seconds
mbapy-cli till process 12345 --check-interval 5
```

### cpu - Wait until CPU utilization drops

Wait until CPU utilization drops below specified threshold.

**Usage:**
```bash
mbapy-cli till cpu <max-utilization> [options]
```

**Arguments:**
- `max-utilization`: Maximum CPU utilization percentage

**Options:**
- `--check-interval`: Check interval in seconds (default: 30)
- `--wait-iter`: Wait iterations after condition met (default: 5)

**Examples:**
```bash
# Wait until CPU utilization drops below 20%
mbapy-cli till cpu 20

# Wait with custom settings
mbapy-cli till cpu 30 --check-interval 60 --wait-iter 3
```

### mem - Wait until system memory is available

Wait until specified amount of system memory becomes available.

**Usage:**
```bash
mbapy-cli till mem <memory> [options]
```

**Arguments:**
- `memory`: Required system memory in GB

**Options:**
- `--check-interval`: Check interval in seconds (default: 30)
- `--wait-iter`: Wait iterations after condition met (default: 5)

**Examples:**
```bash
# Wait until 4GB system memory is available
mbapy-cli till mem 4

# Wait for 8GB with custom settings
mbapy-cli till mem 8 --check-interval 60 --wait-iter 10
```

### file - Wait until file or folder exists

Wait until a specific file or folder appears at the specified path.

**Usage:**
```bash
mbapy-cli till file <path> [options]
```

**Arguments:**
- `path`: Path to file or folder to wait for

**Options:**
- `--check-interval`: Check interval in seconds (default: 10)

**Examples:**
```bash
# Wait for file to appear
mbapy-cli till file /path/to/file.txt

# Wait for folder to appear
mbapy-cli till file /path/to/folder

# Check every 5 seconds
mbapy-cli till file /path/to/file.txt --check-interval 5
```

### folder - Wait until folder contains minimum number of files

Wait until a folder contains at least the specified number of files (non-recursive).

**Usage:**
```bash
mbapy-cli till folder <path> <min-count> [options]
```

**Arguments:**
- `path`: Path to folder to check
- `min-count`: Minimum number of files required

**Options:**
- `--check-interval`: Check interval in seconds (default: 10)

**Examples:**
```bash
# Wait until folder has at least 10 files
mbapy-cli till folder /path/to/folder 10

# Wait with custom check interval
mbapy-cli till folder /path/to/folder 5 --check-interval 5
```

## How It Works

### Time-based waiting
- Uses `tqdm` progress bar for visual feedback
- Supports flexible time format parsing
- Accurate timing with Python's `time.sleep()`

### Resource monitoring (GPU/CPU/Memory)
- GPU monitoring uses `nvidia-smi` command
- CPU and memory monitoring uses `psutil` library
- Real-time progress display with current values
- Configurable check intervals and wait iterations

### Process monitoring
- Uses `psutil` to check process existence
- Configurable check intervals
- Real-time status updates

### File system monitoring
- Uses `os.path.exists()` for file/folder existence
- Non-recursive file counting for folder monitoring
- Configurable check intervals

## Common Use Cases

### Task scheduling
```bash
# Wait until 2 PM to start processing
mbapy-cli till start-time "2026-02-05 14:00:00"
# Then run your task
python process_data.py
```

### Resource management
```bash
# Wait for GPU resources before training
mbapy-cli till cuda-mem 8 --wait-iter 10
# Then start training
python train_model.py
```

### Process synchronization
```bash
# Wait for data processing to complete
mbapy-cli till process 12345
# Then start analysis
python analyze_results.py
```

### File system monitoring
```bash
# Wait for data files to be generated
mbapy-cli till folder /data/output 100
# Then process the files
python batch_process.py
```

## Notes

- All monitoring commands show real-time progress
- GPU monitoring requires NVIDIA GPU with `nvidia-smi` installed
- Process monitoring requires the process to be running
- File system monitoring works with both files and directories
- Time parsing supports flexible unit combinations
- All commands can be interrupted with Ctrl+C

## Dependencies

- `psutil` - System and process monitoring
- `tqdm` - Progress bar display
- `nvidia-smi` (for GPU monitoring) - NVIDIA system management interface

## See Also

- [batch_cmd](batch_cmd.md) - Batch command execution