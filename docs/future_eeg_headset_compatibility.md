# Future EEG headset compatibility

NeuroSense does not currently provide live headset seizure inference. Before integrating a device, confirm all of the following from its documentation and a local Ubuntu test:

- Access to raw, unfiltered EEG samples and their physical units, without a vendor-only seizure score substituting for signals.
- Stable channel labels and electrode placement/reference details that can be mapped to the model's documented montage.
- Sampling rate, timestamp behavior, packet-loss reporting, and synchronization details.
- A maintained Ubuntu-compatible SDK or a supported BrainFlow, Lab Streaming Layer (LSL), or equivalent interface.
- Signal-quality indicators, electrode contact/impedance status where available, and explicit handling of missing or saturated channels.
- Device and SDK licensing, privacy, and data-retention requirements.

Hardware compatibility is separate from CHB-MIT model validity. A model trained on CHB-MIT must not be assumed to generalize to a consumer headset, a different montage, or a different age population. Any future live inference requires its own authorized engineering, validation, and safety review. This document makes no clinical or emergency-alert claim.
