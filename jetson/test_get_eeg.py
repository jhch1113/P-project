from pylsl import StreamInlet, resolve_byprop
import numpy as np

# 1. Find LSL stream (Muse EEG data only)
print("Looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)

if not streams:
    print("No streams found. Is 'muselsl stream' running?")
else:
    inlet = StreamInlet(streams[0])
    try:
        while True:
            # Pull a single data sample
            sample, timestamp = inlet.pull_sample()
            if sample:
                # Print 4-electrode data (TP9, AF7, AF8, TP10)
                print(f"Timestamp: {timestamp} | EEG: {sample[:4]}")
    except KeyboardInterrupt:
        print("Stream stopped.")
