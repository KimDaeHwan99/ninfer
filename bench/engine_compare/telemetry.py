#!/usr/bin/env python3
"""Sample the machine once a second until stopped: per GPU power, temperature, VRAM, utilisation and clocks
(nvidia-smi), CPU temperature (k10temp Tctl), DDR5 module temperatures (spd5118), used system RAM. The current phase
label is read from PHASE_FILE (the benchmark driver writes it), so samples can be grouped per engine and phase.

    python3 telemetry.py OUT.csv PHASE_FILE
"""
import csv
import glob
import subprocess
import sys
import time
from pathlib import Path

out, phase_file = Path(sys.argv[1]), Path(sys.argv[2])
QUERY = "index,power.draw,temperature.gpu,memory.used,utilization.gpu,clocks.sm,clocks.mem"


def hwmon(name):
    """Temperatures in degrees C of every hwmon device with this name (temp*_input of each)."""
    values = []
    for device in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if Path(device, "name").read_text().strip() != name:
                continue
            for sensor in sorted(glob.glob(device + "/temp*_input"))[:1]:
                values.append(int(Path(sensor).read_text()) / 1000)
        except OSError:
            continue
    return values


def ram_used_gib():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    kib = lambda key: int(info[key].split()[0])
    return round((kib("MemTotal") - kib("MemAvailable")) / 1048576, 2)


with out.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["t", "phase", "gpu", "power_w", "temp_c", "vram_mib", "util_pct", "sm_mhz", "mem_mhz",
                     "cpu_c", "dimm_c_max", "ram_used_gib"])
    while True:
        started = time.monotonic()
        phase = phase_file.read_text().strip() if phase_file.exists() else ""
        smi = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        cpu, dimm, ram = hwmon("k10temp"), hwmon("spd5118"), ram_used_gib()
        now = round(time.time(), 1)
        for line in smi:
            cells = [cell.strip() for cell in line.split(",")]
            writer.writerow([now, phase, *cells, cpu[0] if cpu else "", max(dimm) if dimm else "", ram])
        handle.flush()
        time.sleep(max(0.0, 1.0 - (time.monotonic() - started)))
