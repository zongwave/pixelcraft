import re
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import mplcursors
import numpy as np

# Function to parse timestamp (for both Paddle and custom logs)
def parse_timestamp(ts_str):
    if ts_str.startswith('I'):
        # Parse Paddle log timestamp (e.g., I0228 12:28:58.321350 or I0307 01:48:49.906819)
        ts_match = re.match(r"I(\d{4} \d{2}:\d{2}:\d{2}\.\d+)", ts_str)
        if not ts_match:
            raise ValueError(f"Invalid Paddle timestamp format: {ts_str}")
        ts = ts_match.group(1)
        date_part, micro_part = ts.split('.')
        micro_part = micro_part.ljust(6, '0')[:6]
        full_ts = f"2025-{date_part}.{micro_part}"
        return datetime.strptime(full_ts, "%Y-%m%d %H:%M:%S.%f")
    else:
        # Try to parse as a datetime string (e.g., "2025-02-28 14:19:19.802" or "2025-02-28 14:19:19,802")
        try:
            # Try with dot (.) as decimal separator
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                # Try with comma (,) as decimal separator, replacing comma with dot
                ts_str_with_dot = ts_str.replace(',', '.')
                return datetime.strptime(ts_str_with_dot, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                pass

        # Try to match a flexible timestamp format (HH:MM:SS.microseconds, e.g., "07:38:00.575320")
        ts_match = re.match(r"(\d{2}:\d{2}:\d{2}\.\d+)", ts_str)
        if ts_match:
            time_str = ts_match.group(1)
            # Assume current year (2025) and March 2 (based on static_test.log context)
            return datetime.strptime(f"2025-03-02 {time_str}", "%Y-%m-%d %H:%M:%S.%f")
        
        # Parse array format [year, month, day, hour, minute, second, microsecond] (old format)
        try:
            parts = eval(ts_str)  # Safely parse the list [2025, 3, 2, 7, 32, 25, 164119]
            if len(parts) == 7:
                year, month, day, hour, minute, second, microsecond = parts
                return datetime(year, month, day, hour, minute, second, microsecond)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid timestamp format or array format: {ts_str}, error: {e}")
        raise ValueError(f"Invalid custom timestamp format: {ts_str}")

# Function to parse memory log with binary-safe reading
def parse_memory_log(log_file):
    if not log_file:
        print("No memory log file provided, skipping memory parsing.")
        return []

    print(f"Parsing memory log file: {log_file}")
    alloc_pattern = re.compile(r"I\d{4} \d{2}:\d{2}:\d{2}\.\d+ \d+ auto_growth_best_fit_allocator\.cc:169\] Alloc (\d+) bytes, ptr = (0x[0-9a-f]+)")
    free_pattern = re.compile(r"I\d{4} \d{2}:\d{2}:\d{2}\.\d+ \d+ auto_growth_best_fit_allocator\.cc:177\] Free (\d+) bytes, ptr = (0x[0-9a-f]+)")

    memory_events = []

    with open(log_file, 'rb') as f:
        for line in f:
            try:
                line_str = line.decode('utf-8', errors='ignore').strip()
            except UnicodeDecodeError:
                line_str = line.decode('latin1', errors='ignore').strip()
            
            alloc_match = alloc_pattern.match(line_str)
            if alloc_match:
                size = int(alloc_match.group(1))
                ptr = alloc_match.group(2)
                timestamp = parse_timestamp(line_str)
                memory_events.append((timestamp, 'alloc', size, ptr))
                print(f"Parsed memory alloc event - Timestamp: {timestamp}, Size: {size} bytes, Ptr: {ptr}")
                continue

            free_match = free_pattern.match(line_str)
            if free_match:
                size = int(free_match.group(1))
                ptr = free_match.group(2)
                timestamp = parse_timestamp(line_str)
                memory_events.append((timestamp, 'free', size, ptr))
                print(f"Parsed memory free event - Timestamp: {timestamp}, Size: {size} bytes, Ptr: {ptr}")

    print(f"Completed parsing memory log. Found {len(memory_events)} memory events.")
    return sorted(memory_events, key=lambda x: x[0])

# Function to parse HPU memory usage log
def parse_hpu_memory_log(log_file):
    if not log_file:
        print("No log file provided, skipping HPU memory parsing.")
        return []

    print(f"Parsing HPU memory log file: {log_file}")
    memory_pattern = re.compile(
        r"I\d{4} \d{2}:\d{2}:\d{2}\.\d+ \d+ hpu_operator\.cc:\d+\] HPU Memory Usage: (\d+\.\d+)% \(free: (\d+), total: (\d+)\)"
    )

    hpu_memory_events = []

    with open(log_file, 'rb') as f:
        for line in f:
            try:
                line_str = line.decode('utf-8', errors='ignore').strip()
            except UnicodeDecodeError:
                line_str = line.decode('latin1', errors='ignore').strip()
            
            memory_match = memory_pattern.match(line_str)
            if memory_match:
                percentage = float(memory_match.group(1))
                free_bytes = int(memory_match.group(2))
                total_bytes = int(memory_match.group(3))
                used_bytes = total_bytes - free_bytes
                timestamp = parse_timestamp(line_str)
                hpu_memory_events.append((timestamp, percentage, used_bytes, free_bytes, total_bytes))
                print(f"Parsed HPU memory event - Timestamp: {timestamp}, Usage: {percentage}%, Used: {used_bytes} bytes")

    print(f"Completed parsing HPU memory log. Found {len(hpu_memory_events)} HPU memory events.")
    return sorted(hpu_memory_events, key=lambda x: x[0])

# Function to parse transformer operations log with binary-safe reading for specific keywords
def parse_transformer_log(log_file_or_string, is_static=False):
    if not log_file_or_string:
        return []

    # Patterns for specific keywords (dynamic mode)
    init_kv_cache_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) FusedMultiTransformerHPU  Initializing `cache_kvs` with zeros")
    loading_weights_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\] .*Loading weights file")
    loading_config_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d{3})?)\] .*Loading configuration file .*")

    transformer_events = []

    if isinstance(log_file_or_string, str) and not log_file_or_string.endswith('.log'):
        # If log_file_or_string is a string, process it directly
        lines = log_file_or_string.splitlines()
        last_config_line = None  # Initialize for dynamic mode
        current_lines = []

        for line in lines:
            original_line = line
            if original_line.strip().startswith("Variable: var"):
                if current_lines:
                    timestamp = parse_static_event_timestamp(current_lines, is_static)
                    if timestamp is not None and isinstance(timestamp, datetime):
                        event_type = parse_event_type(current_lines)
                        if event_type:
                            transformer_events.append((timestamp, event_type))
                current_lines = [original_line]
            elif current_lines and original_line.strip():
                current_lines.append(original_line)

        # Process the last event if exists
        if current_lines:
            timestamp = parse_static_event_timestamp(current_lines, is_static)
            if timestamp is not None and isinstance(timestamp, datetime):
                event_type = parse_event_type(current_lines)
                if event_type:
                    transformer_events.append((timestamp, event_type))

        # For dynamic mode, add the last "Loading configuration file" as "Loading weights" if it exists
        if not is_static:
            if last_config_line:
                timestamp = last_config_line[0]
                transformer_events.append((timestamp, "Loading weights"))

            # Parse dynamic mode lines
            for line in lines:
                original_line = line
                line = line.strip()
                init_match = init_kv_cache_pattern.match(line)
                if init_match:
                    timestamp = parse_timestamp(init_match.group(1))
                    transformer_events.append((timestamp, "init kv cache"))
                    continue

                loading_weights_match = loading_weights_pattern.search(line)
                if loading_weights_match:
                    timestamp = parse_timestamp(loading_weights_match.group(1))
                    transformer_events.append((timestamp, "Loading weights"))
                    continue

                # Track the last "Loading configuration file" for dynamic mode
                config_match = loading_config_pattern.search(line)
                if config_match:
                    timestamp = parse_timestamp(config_match.group(1))
                    last_config_line = (timestamp, "Loading configuration file")

    else:
        # If log_file_or_string is a file path, read it in binary mode
        with open(log_file_or_string, 'rb') as f:
            current_lines = []
            last_config_line = None  # Initialize for dynamic mode
            for line in f:
                try:
                    original_line = line.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    original_line = line.decode('latin1', errors='ignore')
                if original_line.strip().startswith("Variable: var"):
                    if current_lines:
                        timestamp = parse_static_event_timestamp(current_lines, is_static)
                        if timestamp is not None and isinstance(timestamp, datetime):
                            event_type = parse_event_type(current_lines)
                            if event_type:
                                transformer_events.append((timestamp, event_type))
                    current_lines = [original_line]
                elif current_lines and original_line.strip():
                    current_lines.append(original_line)

            # Process the last event if exists
            if current_lines:
                timestamp = parse_static_event_timestamp(current_lines, is_static)
                if timestamp is not None and isinstance(timestamp, datetime):
                    event_type = parse_event_type(current_lines)
                    if event_type:
                        transformer_events.append((timestamp, event_type))

            # For dynamic mode, add the last "Loading configuration file" as "Loading weights" if it exists
            if not is_static:
                if last_config_line:
                    timestamp = last_config_line[0]
                    transformer_events.append((timestamp, "Loading weights"))

                # Read lines for dynamic mode parsing
                lines = []
                f.seek(0)  # Reset file pointer to beginning
                for line in f:
                    try:
                        lines.append(line.decode('utf-8', errors='ignore').strip())
                    except UnicodeDecodeError:
                        lines.append(line.decode('latin1', errors='ignore').strip())

                # Parse dynamic mode lines
                for line in lines:
                    original_line = line
                    line = line.strip()
                    init_match = init_kv_cache_pattern.match(line)
                    if init_match:
                        timestamp = parse_timestamp(init_match.group(1))
                        transformer_events.append((timestamp, "init kv cache"))
                        continue

                    loading_weights_match = loading_weights_pattern.search(line)
                    if loading_weights_match:
                        timestamp = parse_timestamp(loading_weights_match.group(1))
                        transformer_events.append((timestamp, "Loading weights"))
                        continue

                    # Track the last "Loading configuration file" for dynamic mode
                    config_match = loading_config_pattern.search(line)
                    if config_match:
                        timestamp = parse_timestamp(config_match.group(1))
                        last_config_line = (timestamp, "Loading configuration file")

    return sorted(transformer_events, key=lambda x: x[0])

# Helper function to parse event type from static mode lines (simplified for static logs)
def parse_event_type(lines):
    if len(lines) > 1 and lines[1].startswith("  - message:"):
        message = lines[1].split("message:")[1].strip()
        if "FusedMultiTransformerHPU:_forward_" in message:
            return "cal atten"
        elif "FusedMultiTransformerHPU:_post_process_" in message:
            return "new token"
    return None

# Helper function to parse timestamp from static mode event lines
def parse_static_event_timestamp(lines, is_static):
    if not is_static:
        return None
    
    if len(lines) < 8:
        return None
    
    event_type = None
    if lines[1].startswith("  - message:"):
        message = lines[1].split("message:")[1].strip()
        if "FusedMultiTransformerHPU:_forward_" in message:
            event_type = "cal atten"
        elif "FusedMultiTransformerHPU:_post_process_" in message:
            event_type = "new token"
    
    if not event_type:
        return None
    
    if lines[7].startswith("  - data:"):
        data_str = lines[7].split("data:")[1].strip()
        try:
            time_match = re.search(r"(\d{2}:\d{2}:\d{2}\.\d+)", data_str)
            if time_match:
                time_str = time_match.group(1)
                return datetime.strptime(f"2025-03-02 {time_str}", "%Y-%m-%d %H:%M:%S.%f")
            else:
                data_str = data_str.replace(" ", ",")
                parts = eval(data_str)
                if len(parts) == 7:
                    year, month, day, hour, minute, second, microsecond = parts
                    return datetime(year, month, day, hour, minute, second, microsecond)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid data format in static mode: {data_str}, error: {e}")
    
    return None

# Function to sample transformer events
def sample_transformer_events(transformer_events, interval=32):
    if not transformer_events:
        return []

    relevant_events = [event for event in transformer_events if event[1] in ["init kv cache", "cal atten", "new token"]]
    sampled_events = []

    first_cal_atten = None
    first_gen_new_token = None

    for event in relevant_events:
        if event[1] == "init kv cache":
            sampled_events.append(event)
        elif event[1] == "cal atten" and first_cal_atten is None:
            first_cal_atten = event
            sampled_events.append(event)
        elif event[1] == "new token" and first_gen_new_token is None:
            first_gen_new_token = event
            sampled_events.append(event)
        else:
            if event[1] == "new token":
                sampled_events.append(event)

    gen_new_token_events = [event for event in relevant_events if event[1] == "new token"]
    if gen_new_token_events:
        sampled_gen_new_token = [gen_new_token_events[0]]
        remaining_gen_new_token = gen_new_token_events[1:]
        sampled_remaining = remaining_gen_new_token[::interval] if remaining_gen_new_token else []
        sampled_gen_new_token.extend(sampled_remaining)
        sampled_events = [event for event in sampled_events if event[1] != "new token"]
        sampled_events.extend(sampled_gen_new_token)

    return sorted(sampled_events, key=lambda x: x[0])

# Function to calculate memory metrics
def calculate_memory_metrics(memory_events):
    if not memory_events:
        return [], [], [], []

    timestamps = []
    memory_in_use = []
    alloc_cumulative = []
    free_cumulative = []
    current_usage = 0
    total_alloc = 0
    total_free = 0

    for event in memory_events:
        timestamp, event_type, size, ptr = event
        timestamps.append(timestamp)

        if event_type == 'alloc':
            current_usage += size
            total_alloc += size
        elif event_type == 'free':
            current_usage -= size
            total_free += size

        memory_in_use.append(max(current_usage, 0))
        alloc_cumulative.append(total_alloc)
        free_cumulative.append(total_free)

    return timestamps, memory_in_use, alloc_cumulative, free_cumulative

# Function to calculate model weights and KV cache memory consumption
def calculate_memory_consumption(log_file):
    if not log_file:
        return 0, 0

    match = re.match(r"(?:dynamic|static)_bsz_(\d+)_max_len_(\d+)\.log", log_file)
    if not match:
        return 0, 0

    batch_size = int(match.group(1))
    max_seq_len = int(match.group(2))

    num_params = 7_000_000_000
    bytes_per_param = 2
    model_weights_bytes = num_params * bytes_per_param
    model_weights_gb = model_weights_bytes / (1024**3)

    num_layers = 32
    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads

    kv_cache_per_token_per_layer = 2 * hidden_size * bytes_per_param
    kv_cache_per_token = kv_cache_per_token_per_layer * num_layers
    kv_cache_bytes = batch_size * max_seq_len * kv_cache_per_token
    kv_cache_gb = kv_cache_bytes / (1024**3)

    return model_weights_gb, kv_cache_gb

# Function to plot memory metrics, HPU memory usage, and sampled transformer operations on a single timeline
def plot_timeline(memory_events, transformer_events, hpu_memory_events, memory_log_file=None, transformer_log_file=None, combined_log_file=None):
    model_weights_gb, kv_cache_gb = 0, 0
    if memory_log_file:
        model_weights_gb, kv_cache_gb = calculate_memory_consumption(memory_log_file)
    elif combined_log_file:
        model_weights_gb, kv_cache_gb = calculate_memory_consumption(combined_log_file)

    is_static = memory_log_file and memory_log_file.startswith("static_") if memory_log_file else combined_log_file and combined_log_file.startswith("static_")

    all_timestamps = []
    if memory_events:
        all_timestamps.extend([event[0] for event in memory_events])
    if transformer_events:
        all_timestamps.extend([event[0] for event in transformer_events])
    if hpu_memory_events:
        all_timestamps.extend([event[0] for event in hpu_memory_events])

    if not all_timestamps:
        return

    earliest_time = min(all_timestamps)

    # Process allocator memory events
    memory_timestamps, memory_in_use, alloc_cumulative, free_cumulative = calculate_memory_metrics(memory_events)
    if memory_timestamps:
        relative_memory_times = [(t - earliest_time).total_seconds() for t in memory_timestamps]
        absolute_memory_times = memory_timestamps
    else:
        relative_memory_times = []
        absolute_memory_times = []

    # Process transformer events
    relative_transformer_times = []
    absolute_transformer_times = []
    transformer_labels = []
    if transformer_events:
        for event in transformer_events:
            relative_time = (event[0] - earliest_time).total_seconds()
            relative_transformer_times.append(relative_time)
            absolute_transformer_times.append(event[0])
            transformer_labels.append(event[1])

    sampled_transformer_events = sample_transformer_events(transformer_events, interval=32)
    sampled_relative_times = []
    sampled_absolute_times = []
    sampled_transformer_labels = []
    if sampled_transformer_events:
        for event in sampled_transformer_events:
            relative_time = (event[0] - earliest_time).total_seconds()
            sampled_relative_times.append(relative_time)
            sampled_absolute_times.append(event[0])
            sampled_transformer_labels.append(event[1])

    # Process HPU memory events (only if events exist)
    relative_hpu_times = []
    absolute_hpu_times = []
    hpu_used_gb = []
    hpu_total_gb = 0
    if hpu_memory_events:
        hpu_timestamps = [event[0] for event in hpu_memory_events]
        hpu_used_bytes = [event[2] for event in hpu_memory_events]
        hpu_total_bytes = hpu_memory_events[0][4]  # Total bytes should be constant
        relative_hpu_times = [(t - earliest_time).total_seconds() for t in hpu_timestamps]
        absolute_hpu_times = hpu_timestamps
        hpu_used_gb = [bytes / (1024**3) for bytes in hpu_used_bytes]
        hpu_total_gb = hpu_total_bytes / (1024**3)

    # Convert allocator memory to GB
    memory_in_use_gb = [value / (1024**3) for value in memory_in_use] if memory_in_use else []
    alloc_gb = [value / (1024**3) for value in alloc_cumulative] if alloc_cumulative else []
    free_gb = [value / (1024**3) for value in free_cumulative] if free_cumulative else []

    # Calculate maximums for allocator memory
    max_in_use = max(memory_in_use_gb) if memory_in_use_gb else 0
    max_alloc = max(alloc_gb) if alloc_gb else 0
    max_free = max(free_gb) if free_gb else 0
    max_in_use_idx = memory_in_use_gb.index(max_in_use) if memory_in_use_gb and max_in_use > 0 else 0
    max_alloc_idx = alloc_gb.index(max_alloc) if alloc_gb and max_alloc > 0 else 0
    max_free_idx = free_gb.index(max_free) if free_gb and max_free > 0 else 0

    # Calculate maximum for HPU memory (only if events exist)
    max_hpu_used = max(hpu_used_gb) if hpu_used_gb else 0
    max_hpu_idx = hpu_used_gb.index(max_hpu_used) if hpu_used_gb and max_hpu_used > 0 else 0

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot model weights and KV cache lines
    if model_weights_gb > 0 and kv_cache_gb > 0:
        ax1.axhline(y=model_weights_gb, color='purple', linestyle='--', linewidth=2, label=f'Model Weights ({model_weights_gb:.2f} GB)')
        ax1.axhline(y=kv_cache_gb, color='magenta', linestyle='--', linewidth=2, label=f'KV Cache ({kv_cache_gb:.2f} GB)')
        total_memory_gb = model_weights_gb + kv_cache_gb
        ax1.axhline(y=total_memory_gb, color='black', linestyle='--', linewidth=2, label=f'Total Memory ({total_memory_gb:.2f} GB)')

    # Plot HPU total memory line only if HPU events exist
    if hpu_memory_events and hpu_total_gb > 0:
        ax1.axhline(y=hpu_total_gb, color='red', linestyle='--', linewidth=2, label=f'HPU Total Memory ({hpu_total_gb:.2f} GB)')

    # Plot allocator memory in use
    if memory_in_use:
        memory_in_use_line = ax1.plot(relative_memory_times, memory_in_use_gb, label='Memory In Use (GB)', color='blue', linewidth=3)[0]
        if max_in_use > 0:
            max_relative_time = relative_memory_times[max_in_use_idx]
            ax1.plot(max_relative_time, max_in_use, 'bo', label=f'Max In Use: {max_in_use:.2f} GB', markersize=15)
            ax1.annotate(f'Max In Use: {max_in_use:.2f} GB', 
                         xy=(max_relative_time, max_in_use), 
                         xytext=(10, 10), textcoords='offset points',
                         ha='left', va='bottom', fontsize=10, color='blue', 
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    # Plot cumulative allocations
    if alloc_cumulative:
        alloc_line = ax1.plot(relative_memory_times, alloc_gb, label='Cumulative Allocations (GB)', color='red', linestyle='dotted', linewidth=2)[0]
        if max_alloc > 0:
            max_relative_time = relative_memory_times[max_alloc_idx]
            ax1.plot(max_relative_time, max_alloc, 'ro', label=f'Max Alloc: {max_alloc:.2f} GB', markersize=15)
            ax1.annotate(f'Max Alloc: {max_alloc:.2f} GB', 
                         xy=(max_relative_time, max_alloc), 
                         xytext=(10, 10), textcoords='offset points',
                         ha='left', va='bottom', fontsize=8, color='red')

    # Plot cumulative frees
    if free_cumulative:
        free_line = ax1.plot(relative_memory_times, free_gb, label='Cumulative Frees (GB)', color='green', linestyle='dotted', linewidth=2)[0]
        if max_free > 0:
            max_relative_time = relative_memory_times[max_free_idx]
            ax1.plot(max_relative_time, max_free, 'go', label=f'Max Free: {max_free:.2f} GB', markersize=15)
            ax1.annotate(f'Max Free: {max_free:.2f} GB', 
                         xy=(max_relative_time, max_free), 
                         xytext=(10, 10), textcoords='offset points',
                         ha='left', va='bottom', fontsize=8, color='green')

    # Plot HPU memory usage only if HPU events exist
    if hpu_memory_events and hpu_used_gb:
        hpu_line = ax1.plot(relative_hpu_times, hpu_used_gb, label='HPU Memory Used (GB)', color='orange', linewidth=3)[0]
        if max_hpu_used > 0:
            max_relative_hpu_time = relative_hpu_times[max_hpu_idx]
            ax1.plot(max_relative_hpu_time, max_hpu_used, 'o', color='orange', label=f'Max HPU Used: {max_hpu_used:.2f} GB', markersize=15)
            ax1.annotate(f'Max HPU Used: {max_hpu_used:.2f} GB', 
                         xy=(max_relative_hpu_time, max_hpu_used), 
                         xytext=(10, 10), textcoords='offset points',
                         ha='left', va='bottom', fontsize=10, color='orange',
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    # Plot transformer events
    if sampled_transformer_events:
        transformer_timestamps = sampled_relative_times
        transformer_absolute_times = sampled_absolute_times
        transformer_labels = sampled_transformer_labels
        
        init_kv_cache_timestamps = [ts for ts, label in zip(transformer_timestamps, transformer_labels) if label == "init kv cache"]
        cal_atten_timestamps = [ts for ts, label in zip(transformer_timestamps, transformer_labels) if label == "cal atten"]
        gen_new_token_timestamps = [ts for ts, label in zip(transformer_timestamps, transformer_labels) if label == "new token"]
        
        if init_kv_cache_timestamps:
            init_kv_cache_absolute = [absolute_transformer_times[sampled_relative_times.index(ts)] for ts in init_kv_cache_timestamps]
            ax1.plot(init_kv_cache_timestamps, [0] * len(init_kv_cache_timestamps), 'o', color='purple', markersize=10, label='init kv cache')
            for rel_ts, abs_ts in zip(init_kv_cache_timestamps, init_kv_cache_absolute):
                ax1.annotate(f"init kv cache", 
                             xy=(rel_ts, 0), xytext=(0, -15), textcoords='offset points',
                             rotation=45, ha='right', va='top', fontsize=8, color='black')
        if cal_atten_timestamps and len(cal_atten_timestamps) > 0:
            cal_atten_absolute = [absolute_transformer_times[sampled_relative_times.index(cal_atten_timestamps[0])]]
            ax1.plot([cal_atten_timestamps[0]], [0], 'o', color='green', markersize=10, label='cal atten')
            ax1.annotate(f"cal atten", 
                         xy=(cal_atten_timestamps[0], 0), xytext=(0, -15), textcoords='offset points',
                         rotation=45, ha='right', va='top', fontsize=8, color='black')
        if gen_new_token_timestamps:
            gen_new_token_absolute = [absolute_transformer_times[sampled_relative_times.index(ts)] for ts in gen_new_token_timestamps]
            ax1.plot(gen_new_token_timestamps, [0] * len(gen_new_token_timestamps), 'o', color='cyan', markersize=10, label='new token')
            for rel_ts, abs_ts in zip(gen_new_token_timestamps, gen_new_token_absolute):
                label = "new token"
                ax1.annotate(f"{label}", 
                             xy=(rel_ts, 0), xytext=(0, -15), textcoords='offset points',
                             rotation=45, ha='right', va='top', fontsize=8, color='black')

    loading_weights_relative_timestamps = [(event[0] - earliest_time).total_seconds() for event in transformer_events if event[1] == "Loading weights"]
    if loading_weights_relative_timestamps:
        ax1.plot(loading_weights_relative_timestamps, [0] * len(loading_weights_relative_timestamps), 'o', color='brown', markersize=10, label='Loading weights')
        for rel_ts, abs_ts, label in zip(loading_weights_relative_timestamps, [event[0] for event in transformer_events if event[1] == "Loading weights"], [event[1] for event in transformer_events if event[1] == "Loading weights"]):
            ax1.annotate(f"{label}", 
                         xy=(rel_ts, 0), xytext=(0, -15), textcoords='offset points',
                         rotation=45, ha='right', va='top', fontsize=8, color='black')

    ax1.set_xlabel('Absolute Time')
    ax1.set_ylabel('Memory (GB)')
    ax1.set_title(f'Memory, HPU Usage, and Sampled Transformer Events Timeline ({memory_log_file or "No Memory Log"} / {transformer_log_file or "No Transformer Log"} / {combined_log_file or "No Combined Log"})')

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Relative Time (seconds)')
    plt.setp(ax2.get_xticklabels(), rotation=45)

    # Combine all relative times for tick synchronization
    all_relative_times = relative_memory_times + sampled_relative_times + relative_hpu_times if relative_memory_times or sampled_relative_times or relative_hpu_times else []
    if all_relative_times:
        unique_relative_times = sorted(set(all_relative_times))
        if len(unique_relative_times) > 1:
            interval = (max(unique_relative_times) - min(unique_relative_times)) / 10
            if interval > 0:
                relative_ticks = [min(unique_relative_times) + i * interval for i in range(11)]
                absolute_ticks = []
                all_absolute_times = absolute_memory_times + absolute_transformer_times + absolute_hpu_times
                for rel_tick in relative_ticks:
                    closest_abs_time = min(all_absolute_times,
                                          key=lambda x: abs((x - earliest_time).total_seconds() - rel_tick))
                    absolute_ticks.append(closest_abs_time)
                ax1.set_xticks([(t - earliest_time).total_seconds() for t in absolute_ticks])
                ax1.set_xticklabels([t.strftime('%H:%M:%S.%f')[:-3] for t in absolute_ticks], rotation=45)
                ax2.set_xticks(relative_ticks)
                ax2.set_xticklabels([f'{t:.2f}' for t in relative_ticks], rotation=45)

    ax1.grid(True)
    ax1.legend()

    # Add hover tooltips for all curves
    lines_for_cursor = []
    if memory_in_use:
        lines_for_cursor.append(memory_in_use_line)
    if alloc_cumulative:
        lines_for_cursor.append(alloc_line)
    if free_cumulative:
        lines_for_cursor.append(free_line)
    if hpu_memory_events and hpu_used_gb:
        lines_for_cursor.append(hpu_line)

    if lines_for_cursor:
        cursor = mplcursors.cursor(lines_for_cursor, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            line = sel.artist
            x = sel.target[0]
            if line == memory_in_use_line:
                idx = np.abs(np.array(relative_memory_times) - x).argmin()
                relative_ts = relative_memory_times[idx]
                absolute_ts = absolute_memory_times[idx]
                value = memory_in_use_gb[idx]
                sel.annotation.set_text(f'Relative Time: {relative_ts:.2f} s\nAbsolute Time: {absolute_ts.strftime("%H:%M:%S.%f")[:-3]}\nMemory In Use: {value:.2f} GB')
            elif line == alloc_line:
                idx = np.abs(np.array(relative_memory_times) - x).argmin()
                relative_ts = relative_memory_times[idx]
                absolute_ts = absolute_memory_times[idx]
                value = alloc_gb[idx]
                sel.annotation.set_text(f'Relative Time: {relative_ts:.2f} s\nAbsolute Time: {absolute_ts.strftime("%H:%M:%S.%f")[:-3]}\nCumulative Alloc: {value:.2f} GB')
            elif line == free_line:
                idx = np.abs(np.array(relative_memory_times) - x).argmin()
                relative_ts = relative_memory_times[idx]
                absolute_ts = absolute_memory_times[idx]
                value = free_gb[idx]
                sel.annotation.set_text(f'Relative Time: {relative_ts:.2f} s\nAbsolute Time: {absolute_ts.strftime("%H:%M:%S.%f")[:-3]}\nCumulative Free: {value:.2f} GB')
            elif line == hpu_line:
                idx = np.abs(np.array(relative_hpu_times) - x).argmin()
                relative_ts = relative_hpu_times[idx]
                absolute_ts = absolute_hpu_times[idx]
                value = hpu_used_gb[idx]
                sel.annotation.set_text(f'Relative Time: {relative_ts:.2f} s\nAbsolute Time: {absolute_ts.strftime("%H:%M:%S.%f")[:-3]}\nHPU Memory Used: {value:.2f} GB')

    log_prefix = (memory_log_file or transformer_log_file or combined_log_file or 'no_log')
    if log_prefix:
        log_prefix = log_prefix.split('/')[-1].replace('.log', '')
    output_file = f"{log_prefix}_memory_and_events_timeline.png"
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze memory and sampled transformer events from PaddlePaddle and custom logs.")
    parser.add_argument("--mem", type=str, nargs='?', default=None, help="Path to the memory log file (e.g., dynamic_bsz_10_max_len_512.log)")
    parser.add_argument("--trans", type=str, nargs='?', default=None, help="Path to the transformer operations log file or string")
    parser.add_argument("--log", type=str, nargs='?', default=None, help="Path to a single log file containing both memory and transformer operations")

    args = parser.parse_args()
    memory_log_file = args.mem
    transformer_log = args.trans
    combined_log_file = args.log

    # Parse memory log from --mem or --log
    memory_events = []
    if memory_log_file:
        memory_events = parse_memory_log(memory_log_file)
    elif combined_log_file:
        memory_events = parse_memory_log(combined_log_file)

    # Parse transformer log from --trans or --log
    transformer_events = []
    is_static = memory_log_file and memory_log_file.startswith("static_") if memory_log_file else combined_log_file and combined_log_file.startswith("static_")
    if transformer_log:
        transformer_events = parse_transformer_log(transformer_log, is_static)
    elif combined_log_file:
        transformer_events = parse_transformer_log(combined_log_file, is_static)

    # Parse HPU memory usage from --mem or --log
    hpu_memory_events = []
    if memory_log_file:
        hpu_memory_events = parse_hpu_memory_log(memory_log_file)
    elif combined_log_file:
        hpu_memory_events = parse_hpu_memory_log(combined_log_file)

    # Plot combined timeline
    if memory_events or transformer_events or hpu_memory_events:
        plot_timeline(memory_events, transformer_events, hpu_memory_events, memory_log_file, transformer_log, combined_log_file)
    else:
        print("No events found in the logs. Provide at least one log file or string via --mem, --trans, or --log.")