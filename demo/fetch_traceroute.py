import json
import os
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from ripe.atlas.cousteau import AtlasResultsRequest

# ==========================================
# CONFIGURATION
# ==========================================

MSM_IDS = [
    5009, 5010, 5011, 5012, 5013, 5004, 
    5014, 5015, 5005, 5016, 5001, 5008, 5006
]

PROBE_IDS = [] # All probes

# Strategy: 2 Years, Weekly, 6 times a day
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 1, 1)

# Capture data at these hours (UTC) every collection day
# Covers all global peak/off-peak times
TARGET_HOURS = [0, 4, 8, 12, 16, 20] 

SNAPSHOT_DURATION_MINUTES = 30

OUTPUT_DIR = "traceroutes"

# ==========================================
# PREPROCESSING (Unchanged)
# ==========================================
def clean_traceroute_to_sentence(raw_data):
    src_ip = raw_data.get('src_addr', 'UNKNOWN')
    dst_ip = raw_data.get('dst_addr', 'UNKNOWN')
    proto = raw_data.get('proto', 'UNKNOWN')
    path_tokens = []
    
    if 'result' in raw_data:
        for hop in raw_data['result']:
            hop_ip = "*" 
            if 'result' in hop:
                for packet in hop['result']:
                    if 'from' in packet:
                        hop_ip = packet['from']
                        break 
            path_tokens.append(hop_ip)

    if not path_tokens: return None
    path_str = " ".join(path_tokens)
    return f"[SRC] {src_ip} [DST] {dst_ip} [PROTO] {proto} [PATH] {path_str}"

# ==========================================
# NEW FETCH LOOP (Weekly + Multi-Hour)
# ==========================================
def fetch_and_process():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # Start iterating by WEEK
    current_date = START_DATE

    while current_date < END_DATE:
        print(f"\n=== Processing Week of: {current_date.strftime('%Y-%m-%d')} ===")

        # Inside this day, iterate through the specific HOURS
        for hour in TARGET_HOURS:
            # Set the exact start time for this snapshot
            snapshot_start = current_date.replace(hour=hour, minute=0, second=0)
            snapshot_stop = snapshot_start + timedelta(minutes=SNAPSHOT_DURATION_MINUTES)
            
            print(f"  > Snapshot: {snapshot_start} to {snapshot_stop}")

            for msm_id in MSM_IDS:
                # Directory structure: traceroutes/5006/
                msm_dir = os.path.join(OUTPUT_DIR, str(msm_id))
                if not os.path.exists(msm_dir): os.makedirs(msm_dir)
                
                # Filename includes DATE and HOUR
                # e.g., traceroutes_2023_01_01_h08.txt
                filename = f"traceroutes_{snapshot_start.strftime('%Y_%m_%d_h%H')}.txt"
                filepath = os.path.join(msm_dir, filename)
                
                if os.path.exists(filepath):
                    continue # Skip if done

                kwargs = {
                    "msm_id": msm_id,
                    "start": snapshot_start,
                    "stop": snapshot_stop
                }
                if PROBE_IDS: kwargs["probe_ids"] = PROBE_IDS

                try:
                    is_success, results = AtlasResultsRequest(**kwargs).create()
                    
                    if is_success:
                        # Buffer lines to write fewer times to disk
                        lines_to_write = []
                        for entry in results:
                            sentence = clean_traceroute_to_sentence(entry)
                            if sentence:
                                lines_to_write.append(sentence)
                        
                        # Write to file
                        if lines_to_write:
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write("\n".join(lines_to_write))
                            print(f"      [MSM {msm_id}] Saved {len(lines_to_write)} traces.")
                        else:
                            print(f"      [MSM {msm_id}] No valid traces found.")
                    else:
                        print(f"      [MSM {msm_id}] API returned failure.")
                
                except Exception as e:
                    print(f"      [MSM {msm_id}] Error: {e}")
                
                # Sleep to prevent API Rate Limiting (Important for tight loops)
                time.sleep(0.5)

        # Advance by 1 WEEK
        current_date += timedelta(weeks=1)

if __name__ == "__main__":
    fetch_and_process()