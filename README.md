# AMR Project — Team ABS

Autonomous Mobile Robots, Hochschule Bonn-Rhein-Sieg, SS26.
Robile omnidirectional platform, ROS2 Humble, Ubuntu 22.04.

| Task | Status |
| --- | --- |
| 1. Path and motion planning (A* + potential field) | working, tested on the real Robile |
| 2. Localisation (Monte Carlo / particle filter) | working, tested in simulation |
| 3. Environment exploration | not started |

---

## 1. What this package does

```
                       map_publisher
                             |
                             |  /map (OccupancyGrid, latched)
                             v
  /scan  ------------>  particle_filter  ------------> TF map->odom
  TF odom->base_link         |                              |
                             |  /mcl_pose                   |
                             |  /particle_cloud             |
                                                            v
  /goal_pose  -------->  a_star_planner  <----------------- TF map->base_link
                             |
                             |  /global_path (nav_msgs/Path)
                             v
  /scan  ------------>  potential_field_planner  ------->  /cmd_vel
                                                                |
                                                                v
                                                              Robot
```

Division of responsibility:

- **`map_publisher`** reads the `.pgm`/`.yaml` and publishes `/map` latched.
- **`particle_filter`** estimates where the robot is on that map and
  publishes the `map → odom` correction. This is Task 2.
- **`a_star_planner`** searches the occupancy grid for a route and
  publishes a list of waypoints. It knows about walls from the map but
  nothing about live obstacles.
- **`potential_field_planner`** chases one waypoint at a time and avoids
  whatever the laser sees. It is the only node that publishes velocity.

The key architectural point: the potential field planner never sees the
final goal, and A* never sees a live obstacle. Each does one job.

---

## 2. Repository layout

```
amr-team-abs/
├── amr_team_abs/
│   ├── astar.py                   standalone A* library + CLI, no ROS
│   ├── a_star_planner.py          ROS2 wrapper: /map + /goal_pose -> /global_path
│   ├── potential_field_planner.py local planner + waypoint manager -> /cmd_vel
│   ├── particle_filter.py         Monte Carlo Localisation  [Task 2]
│   ├── map_publisher.py           latched /map without nav2
│   ├── fake_robot_sim.py          headless simulator for testing on the lab map
│   ├── visualise_map.py           offline map inspection (matplotlib)
│   └── test_particle_filter.py    offline unit test for the filter maths
├── launch/
│   └── task1_navigation.launch.py full stack, sim or real
├── config/
│   └── task1.rviz                 RViz layout
├── maps/
│   ├── my_lab_map5new.pgm         SLAM map of the lab
│   └── my_lab_map5new.yaml
├── package.xml
└── setup.py
```

---

## 3. Build

```bash
cd ~/ros2_ws
colcon build --packages-select amr-team-abs --symlink-install
source install/setup.bash
ros2 pkg executables amr-team-abs
```

Four executables should be listed. `--symlink-install` means Python
edits take effect without rebuilding.

Python dependencies: `numpy`, `scipy`, `pyyaml`, `pillow`,
`tf_transformations`.

```bash
pip3 install numpy scipy pyyaml pillow
sudo apt install ros-humble-tf-transformations
```

nav2 is **not** required. `map_publisher` replaces `map_server` and
`particle_filter` replaces `amcl`.

---

## 4. Running

Set the same `ROS_DOMAIN_ID` in every terminal — `0` for simulation, the
robot number on hardware. A mismatch makes `ros2 topic list` come back
nearly empty and nothing discovers anything.

### 4.1 Localisation on the lab map, no Gazebo

There is no Gazebo world matching `my_lab_map5new`, so `fake_robot_sim`
stands in: it holds a ground-truth pose, drives it from `/cmd_vel`,
raycasts the map to produce `/scan`, and publishes odometry with drift.

```bash
# terminal 1
ros2 launch amr-team-abs task1_navigation.launch.py mode:=real planners:=false

# terminal 2
cd ~/ros2_ws/src/amr-team-abs
python3 amr_team_abs/fake_robot_sim.py --ros-args \
  -p map_yaml:=$PWD/maps/my_lab_map5new.yaml \
  -p start_x:=-1.52 -p start_y:=-7.15

# terminal 3
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Because the simulator knows the true pose, it prints the real
localisation error every second:

```
truth (-1.32, -7.15, +0)  mcl (-1.24, -7.09, +1)  err 0.100 m / 1.0 deg
```

### 4.2 Full stack

Drop `planners:=false`, let the filter converge, then send a goal:

```bash
ros2 launch amr-team-abs task1_navigation.launch.py mode:=real
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 3.0, y: -6.0}, orientation: {w: 1.0}}}"
```

Or click **2D Goal Pose** in RViz.

### 4.3 Gazebo, closed_walls

```bash
ros2 launch robile_gazebo gazebo_4_wheel.launch.py x_pose:=3.0 y_pose:=-1.5
ros2 launch amr-team-abs task1_navigation.launch.py mode:=sim
```

### 4.4 The real Robile

Drivers on the robot, inside tmux:

```bash
ssh -x studentkelo@192.168.0.104
tmux new -s bringup
source ~/ros2ws/install/setup.bash
ros2 launch robile_bringup robot.launch.py
```

On the laptop:

```bash
export ROS_DOMAIN_ID=4
ros2 launch amr-team-abs task1_navigation.launch.py mode:=real
```

Set the starting pose with RViz's **2D Pose Estimate**.

### 4.5 Launch arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `mode` | `real` | `sim` uses closed_walls and simulated time; `real` uses the lab map |
| `map` | auto | full path to a different map `.yaml` |
| `init_x/y/yaw` | per map | where the robot is standing |
| `planners` | `true` | `false` = map + localisation only |
| `planner_delay` | `6.0` | seconds before A* and the potential field start |
| `rviz` | `true` | |

---

## 5. Approach

### 5.1 A* global planner

8-connected grid search with an octile heuristic, which is admissible
for 8-connected movement. Obstacles are inflated by the robot radius
using a distance transform, so the planner treats the robot as a point.
Diagonal moves through wall corners are rejected explicitly. The raw
path has one cell per step, far too many waypoints, so `simplify()`
keeps only the cells where direction changes.

`astar.py` is pure Python with no ROS imports, so the algorithm can be
developed and visualised standalone:

```bash
python3 amr_team_abs/astar.py maps/my_lab_map5new.yaml --start -1.5 -7.1 --goal 3.0 -6.0
```

### 5.2 Potential field local planner

An attractive force toward the current waypoint, normalised so its
magnitude does not grow with distance, plus a repulsive force from every
laser return inside `rho_0`. The sum gives a desired heading; a
proportional controller turns that into `/cmd_vel`, with a low-pass
filter on the output so the commands are smooth enough for real motors.

The waypoint manager is folded into the same node: it advances the index
once the robot is within `goal_threshold` of the current waypoint.

### 5.3 Monte Carlo Localisation

The three steps from the lecture:

| Step | Function |
| --- | --- |
| Sample motion model | `sample_motion_omni` / `sample_motion_odometry` |
| Measurement model | `likelihood_field_weights` |
| Resampling | `systematic_resample` |
| Pose estimate | `weighted_mean_pose`, `pose_covariance` |
| Output | `compose_map_to_odom` |
| Adaptive extension | `kld_resample`, `kld_sample_size` |

Four decisions worth explaining:

**Likelihood field instead of the beam model.** The textbook beam model
ray-casts once per particle per beam — 500 × 60 = 30000 casts per
update, far too slow in Python. The likelihood field precomputes a
distance transform of the map once; every beam then becomes a single
array lookup, and the whole update is one vectorised numpy operation.

**An omnidirectional motion model.** The classic rot1/trans/rot2
decomposition assumes the robot can only drive along its own x axis. The
Robile can strafe, so a pure sideways move would be modelled as
turn-90-drive-turn-back and inject nonsense rotational noise. Our `omni`
model perturbs (dx, dy, dθ) in the robot frame directly.
`motion_model: "diff"` selects the textbook version for comparison.

**Weight softening.** Multiplying 60 beam probabilities produces a
distribution so peaked that one particle takes essentially all the
weight and the filter collapses after a single update. Scaling the
summed log-likelihood by 0.3 flattens it. AMCL solves the same problem
by summing `p³` rather than multiplying.

**Resample only when N_eff drops.** Resampling every step destroys
particle diversity for no benefit. We resample only when the effective
sample size falls below half the population.

The node's output is the `map → odom` transform, not a pose. The filter
estimates where `base_link` is in the map, TF already knows where
`base_link` is in `odom`, and `compose_map_to_odom` publishes the
difference. `/odom` therefore stays continuous, and everything
downstream simply asks TF.

**Adaptive extension.** Setting `use_kld_sampling: true` enables KLD
sampling: the number of particles adapts to how uncertain the filter is,
using Fox's bound on the KL divergence between the sampled and true
distributions. In practice it starts near 2000 particles and drops to
around 200 once converged.

---

## 6. Results

Measured with `fake_robot_sim`, which knows the ground truth:

| | |
| --- | --- |
| Initial error | 0.78 m, 20° |
| Error after ~2 s of driving | < 0.20 m |
| Steady-state error | ≈ 0.09 m, < 2° |
| Update rate | 10 Hz, 500 particles, 60 beams |
| KLD particle count | 2000 → 200 after convergence |

`test_particle_filter.py` reproduces these numbers offline with no ROS.

---

## 7. Challenges

**Frame confusion between `odom` and `map`.** The potential field
planner originally looked up `odom → base_link` while chasing waypoints
published in the `map` frame. While `map → odom` was a static identity
this was invisible. The moment localisation started applying a real
correction — about 7 m on our map — the robot began driving toward
coordinates that did not exist, which looked like random wandering. One
line of code, a long time to find.

**Illegal package name.** The package was registered as `amr-team-abs`.
Hyphens are not allowed in ROS2 package names under REP-144, so
`ros2 run amr_team_abs ...` reported "Package not found" while
`colcon build` appeared to succeed. Four separate places have to agree:
`package.xml`, `setup.py`, `setup.cfg`, and the marker file in
`resource/`.

**Unknown space read as free.** Our map `.yaml` had `free_thresh: 0.25`.
The unknown grey that `map_saver` writes is pixel 205, which gives
`occ = (255-205)/255 = 0.196` — below 0.25, and therefore classified as
free. All 16559 unmapped cells became drivable and A* would plan
straight through unexplored space. The correct value is `0.196`, which
is where that odd-looking default in ROS map files comes from.

**Clock skew between robot and laptop.** The robot's clock was about 355
days behind the laptop's. AMCL failed with `Failed to transform initial
pose in time` and would not publish a transform at all. Our own nodes
survived only because they request the latest available transform rather
than one at a specific stamp.

**QoS mismatches.** A `BEST_EFFORT` publisher cannot deliver to a
`RELIABLE` subscriber. Real lidars publish BEST_EFFORT while RViz
defaults to RELIABLE, which produces a silent "no messages will be sent"
warning rather than an error. Diagnose with
`ros2 topic info /scan --verbose`.

**Domain ID discipline.** Every terminal that talks to the robot needs
the same `ROS_DOMAIN_ID`. One robot had `ROS_DOMAIN_ID=3` in its
`.bashrc` while being Robile4, which breaks discovery silently.

**No Gazebo world for the lab map.** `my_lab_map5new` came from SLAM, so
there is no simulator world that matches it. Rather than build one, we
wrote `fake_robot_sim.py`, which raycasts the map directly. This turned
out to be more useful than a Gazebo world would have been, because it
knows the ground truth and can therefore report the actual localisation
error.

**A near-featureless map.** The lab is a long, narrow, rotated rectangle
with a single interior pillar. Two parallel walls constrain position
across the room but barely constrain position along it, so the error
plateaus around 0.09 m rather than approaching zero, and it is larger
along the corridor than across it. The same symmetry means global
localisation frequently locks onto a 180°-flipped hypothesis that fits
the scans equally well. Both are properties of the map, not bugs in the
filter, and driving past the pillar resolves both.

---

## 8. Known limitations

- Global localisation is unreliable on this map (see above). Use RViz's
  **2D Pose Estimate**.
- A 0.40 m inflation radius seals off roughly a third of the free space
  on this map. Narrow doorways may be treated as impassable.
- The particle filter assumes an axis-aligned map origin; a map saved
  with a rotated origin would produce wrong coordinates. `map_publisher`
  warns if it sees one.
- Task 3 (exploration) is not implemented.

---

## 9. Team

| Issue | Scope | Branch |
| --- | --- | --- |
| 1 | Potential field local planner | `issue-1-potential-field` |
| 2 | A* global planner | `Issue-2-Astar-planner` |
| 3 | Integrate global and local planners | `Issue-3-Integrate-Global-and-Local-Planners` |
| 4 | Monte Carlo Localisation | `issue-4-localisation` |
