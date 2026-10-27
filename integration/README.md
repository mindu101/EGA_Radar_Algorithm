# Radar to sentry decision port

The radar process publishes `rm2026_vision_interface/msg/SentryRadarSnapshot` on
`/sentry/radar_snapshot` after each committed frame. The matching consumer edits
are in `sentry_decide_radar_ports.patch` and have also been applied to the local
`27BT-UC/src/sentry_decide` source tree. Apply that patch to an unmodified copy
of the supplied decision project with `patch -p1` from its `27BT-UC` root.

Build `rm2026_vision_interface` first and source its install environment before
starting the radar Python process or building `sentry_decide`. Both sides use
reliable QoS with depth 1. The old one-byte referee `Radar2SentryData` and old
`Radar2Sentry.msg` are independent interfaces and are unchanged.

The snapshot has five fixed enemy slots. Red radar publishes blue IDs
`101,102,103,104,107`; blue radar publishes red IDs `1,2,3,4,7`. Position
values are metres. `known` and `visible` require a confirmed identity and a
measurement from the current frame; guessed and Kalman-only positions remain
unknown. Mate, outpost, and kill fields stay explicitly unknown until a reliable
source is connected. The decision adapter computes freshness locally using the
producer's `source_age_ms` and its own monotonic receipt clock. Targets are
hidden after 300 ms and the link is lost after 1000 ms by default.

Coordinates are labelled `radar_field`. The decision adapter applies the
decision configuration's `field_to_map` transform before handing positions to
the core. The supplied decision configuration currently disables that transform.
Calibrate it against known map points for each side, and set
`enemy_rel_frame: 1`, before using radar positions for live navigation or fire.
The decision node exposes `radar_topic` and `radar_frame_id` ROS parameters.

The standalone `sentry_radar_port_smoke` CMake target compiles the edited
decision core and ROS adapter without the decision project's BehaviorTree
dependency. Source ROS and the interface install environment, then run CMake
with `-DSENTRY_DECIDE_SOURCE_DIR=<path-to-27BT-UC/src/sentry_decide>`.
