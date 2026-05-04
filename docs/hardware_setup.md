# Hardware setup — OpenBCI Cyton + USB dongle

This is the minimum recipe to get the Cyton streaming clean SSVEP into the
real-time pipeline. Plan ~30 minutes the first time.

## 1. USB dongle pairing

The Cyton ships with a small USB radio dongle ("OpenBCI dongle"). Plug it into
your laptop **before** powering the Cyton. Flip the Cyton's switch from `OFF`
through `PC` (battery) — the on-board LED should pulse blue. The dongle's
small switch must be in the `GPIO 6` position (default, factory).

### macOS — find the serial port

```bash
ls /dev/cu.usbserial-*
# typical: /dev/cu.usbserial-DM00DXR8
```

If multiple show up, unplug the dongle, list, replug, list again — the new
entry is yours. Put the path into `config/default.yaml`:

```yaml
acquisition:
  cyton:
    serial_port: /dev/cu.usbserial-DM00DXR8
```

### Linux

`ls /dev/ttyUSB*` — usually `/dev/ttyUSB0`. Add yourself to `dialout`
(`sudo usermod -aG dialout $USER`, then log out/in).

### Windows

Device Manager → Ports (COM & LPT) → "USB Serial Port (COMx)". Use `COMx` as
the `serial_port`.

## 2. Electrode placement (8 channels → posterior occipital strip)

SSVEP signal is strongest over visual cortex. Map Cyton's 8 inputs `N1P..N8P`
to the standard 10-20 occipital ring:

```
            (Cz reference, AFz ground)

           PO7  PO3  POz  PO4  PO8
                O1   Oz   O2

  N1P → PO7   N2P → PO3   N3P → POz   N4P → PO4
  N5P → PO8   N6P → O1    N7P → Oz    N8P → O2

  SRB pin → linked-mastoid reference (clip to A1/A2)
  BIAS pin → ground (forehead / Fpz)
```

The same channel order is in `config/default.yaml > acquisition.channels`,
which the live pipeline uses for filtering and FBCCA/TRCA scoring.

## 3. Impedance check

Before recording, verify per-channel impedance < 20 kΩ (gel) or < 50 kΩ
(dry). The OpenBCI GUI's "Impedance" tab is the easiest path:

1. Open OpenBCI GUI → Cyton (Live, from Cyton) → Start System.
2. Switch to the **Impedance** tab. Click each channel; aim for green.
3. If a channel is red: lift the electrode, abrade the scalp gently, re-gel,
   re-seat. Hair partings matter much more than people expect.

Close the GUI before running the live demo — only one process at a time can
hold the dongle's serial port.

## 4. Stim monitor

PsychoPy will create a window on your default display. Make sure:

- Refresh rate is set to **60 Hz** (System Settings → Displays). Our default
  frequencies (7.5, 8.57, 10, 12 Hz) are integer divisors of 60 Hz, which is
  what makes the flicker phase-stable.
- "Disable display scaling" if you intend to measure reaction times (cosmetic
  for SSVEP).
- Sit ~60 cm from the screen. Each square is 200 px in the default config.

## 5. Quick sanity run

```bash
# 1. activate env, plug dongle, switch Cyton to PC
conda activate ssvep
ls /dev/cu.usbserial-*

# 2. raw stream check (no stim, no UI)
python -m src.apps.live_demo --source cyton --algo fbcca \
       --no-stim --direct --duration 10
```

You should see `[pred] freq=...` lines every ~200 ms. If predictions are stuck
on one frequency before any stim is shown, expect that — without input the
classifier picks whatever band has the most ambient noise.

## 6. Common issues

| symptom | likely cause |
|---|---|
| `Could not open serial port` | wrong path / GUI still running |
| flat-line on one channel | electrode unplugged at the breakout board |
| 60 Hz everywhere | mains pickup — re-seat reference, check ground |
| pred jitters between two adjacent freqs | window too short — try `--config` with `processing.window_s: 3` |
| stim feels uneven | monitor not at 60 Hz, or PsychoPy missed flips — close other apps |
