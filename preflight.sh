#!/usr/bin/env bash
#
# Preflight check for the Robile. Run this BEFORE every launch.
#
#   bash preflight.sh
#
# Checks each link in the chain in order and stops at the first break,
# telling you exactly what to do about it. Five minutes here beats an
# hour of "why is RViz empty".

ROBOT_IP="${ROBOT_IP:-192.168.0.103}"
EXPECT_DOMAIN="${EXPECT_DOMAIN:-3}"

pass() { printf '  \033[0;32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[0;31mFAIL\033[0m  %s\n' "$*"; }
info() { printf '        %s\n' "$*"; }
hdr()  { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }

DIE=0

# ----------------------------------------------------------------------
hdr "1. Environment"
# ----------------------------------------------------------------------
if [ -z "$ROS_DOMAIN_ID" ]; then
    fail "ROS_DOMAIN_ID is not set"
    info "export ROS_DOMAIN_ID=$EXPECT_DOMAIN"
    DIE=1
elif [ "$ROS_DOMAIN_ID" != "$EXPECT_DOMAIN" ]; then
    fail "ROS_DOMAIN_ID is $ROS_DOMAIN_ID, expected $EXPECT_DOMAIN"
    info "It must match the robot number, in EVERY terminal."
    DIE=1
else
    pass "ROS_DOMAIN_ID = $ROS_DOMAIN_ID"
fi

if [ "$RMW_IMPLEMENTATION" != "rmw_fastrtps_cpp" ]; then
    fail "RMW_IMPLEMENTATION = '${RMW_IMPLEMENTATION:-unset}'"
    info "source ~/.bashrc, or export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
    DIE=1
else
    pass "RMW_IMPLEMENTATION = $RMW_IMPLEMENTATION"
fi

[ "$DIE" = "1" ] && { echo; echo "Fix the above, then re-run."; exit 1; }

# ----------------------------------------------------------------------
hdr "2. Can we see the robot?"
# ----------------------------------------------------------------------
TOPICS=$(timeout 8 ros2 topic list 2>/dev/null)
if [ -z "$TOPICS" ]; then
    fail "ros2 topic list is empty"
    info "The drivers are not running, or discovery is broken."
    info "  ssh -x studentkelo@$ROBOT_IP"
    info "  tmux a -t bringup     # is it alive?"
    info "If the session is gone:"
    info "  tmux new -s bringup"
    info "  source ~/ros2ws/install/setup.bash"
    info "  ros2 launch robile_bringup robot.launch.py"
    info "Then on the laptop: ros2 daemon stop && ros2 daemon start"
    exit 1
fi
pass "$(echo "$TOPICS" | wc -l) topics visible"

for t in /scan /odom /cmd_vel /tf; do
    if echo "$TOPICS" | grep -qx "$t"; then
        pass "$t present"
    else
        fail "$t MISSING"
        DIE=1
    fi
done
[ "$DIE" = "1" ] && { echo; echo "Drivers are only partly up. Restart robot.launch.py."; exit 1; }

# ----------------------------------------------------------------------
hdr "3. Is the laser actually publishing?"
# ----------------------------------------------------------------------
HZ=$(timeout 6 ros2 topic hz /scan 2>/dev/null | grep -m1 'average rate' | awk '{print $3}')
if [ -z "$HZ" ]; then
    fail "/scan exists but no messages are arriving"
    info "The lidar driver is not running on the robot."
    exit 1
fi
pass "/scan at ${HZ} Hz"

FRAME=$(timeout 6 ros2 topic echo /scan --once --field header.frame_id 2>/dev/null | head -1)
pass "laser frame = ${FRAME:-unknown}"

RELIABILITY=$(timeout 6 ros2 topic info /scan --verbose 2>/dev/null | grep -m1 Reliability | awk '{print $2}')
pass "/scan reliability = ${RELIABILITY:-unknown}"

# ----------------------------------------------------------------------
hdr "4. CLOCK SKEW  (this is the one that keeps biting)"
# ----------------------------------------------------------------------
SCAN_SEC=$(timeout 6 ros2 topic echo /scan --once --field header.stamp.sec 2>/dev/null | head -1 | tr -d ' -')
NOW=$(date +%s)

if [ -z "$SCAN_SEC" ]; then
    fail "could not read a scan timestamp"
    DIE=1
else
    SKEW=$(( NOW - SCAN_SEC ))
    ABS=${SKEW#-}
    if [ "$ABS" -gt 5 ]; then
        fail "robot clock is ${SKEW}s away from this laptop ($(( ABS / 86400 )) days)"
        info ""
        info "This is why slam_toolbox drops scans, why RViz shows nothing,"
        info "and why A* reports extrapolation errors. FIX IT NOW:"
        info ""
        info "  ssh -x studentkelo@$ROBOT_IP"
        info "  sudo date -s \"$(date '+%Y-%m-%d %H:%M:%S')\""
        info ""
        info "Then restart the drivers:"
        info "  tmux a -t bringup    # Ctrl+C, then relaunch, then Ctrl+b d"
        DIE=1
    else
        pass "clocks agree within ${ABS}s"
    fi
fi

# ----------------------------------------------------------------------
hdr "5. Transforms"
# ----------------------------------------------------------------------
if timeout 6 ros2 run tf2_ros tf2_echo odom base_link 2>/dev/null | grep -q Translation; then
    pass "odom -> base_link exists"
else
    fail "odom -> base_link MISSING - odometry is not publishing"
    DIE=1
fi

if [ -n "$FRAME" ] && timeout 6 ros2 run tf2_ros tf2_echo base_link "$FRAME" 2>/dev/null | grep -q Translation; then
    pass "base_link -> $FRAME exists"
else
    fail "base_link -> $FRAME MISSING - the URDF is not loaded"
    DIE=1
fi

# ----------------------------------------------------------------------
hdr "6. Nothing already owns map->odom"
# ----------------------------------------------------------------------
NODES=$(timeout 6 ros2 node list 2>/dev/null)
CONFLICT=$(echo "$NODES" | grep -E 'particle_filter|slam_toolbox|amcl|static_transform|exploration_mapper')
if [ -n "$CONFLICT" ]; then
    fail "these are already running and will fight over map->odom:"
    echo "$CONFLICT" | sed 's/^/          /'
    info "Kill them before launching."
    DIE=1
else
    pass "map->odom is free"
fi

# ----------------------------------------------------------------------
echo
if [ "$DIE" = "1" ]; then
    printf '\033[0;31mNOT READY\033[0m - fix the FAILs above, then re-run.\n'
    exit 1
fi

printf '\033[0;32mREADY\033[0m\n\n'
echo "Place the robot with ~1 m clear all round (it spins 8 s), then:"
echo "  ros2 launch amr-team-abs task3_exploration.launch.py"
echo
echo "Once slam_toolbox is up, in another terminal:"
echo "  ros2 param set /slam_toolbox minimum_travel_distance 0.3"
