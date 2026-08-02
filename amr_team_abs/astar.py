#!/usr/bin/env python3
"""Standalone A* global planner for a ROS2 map (.pgm + .yaml)."""

import argparse
import heapq
import math
import os

import numpy as np
import yaml
from PIL import Image
import matplotlib.pyplot as plt


def load_map(yaml_path):
    """Load a ROS2 map. Returns (grid, info).

    grid values: 0 = free, 100 = occupied, -1 = unknown

    The .pgm image has row 0 at the TOP, but the ROS map frame puts its
    origin at the BOTTOM-left. We flip vertically here so row 0 = bottom.
    Every conversion below assumes this flip already happened.
    """
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    img_path = meta["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)

    pixels = np.flipud(np.array(Image.open(img_path), dtype=np.float64))

    if int(meta.get("negate", 0)):
        p = pixels / 255.0
    else:
        p = (255.0 - pixels) / 255.0

    grid = np.full(p.shape, -1, dtype=np.int8)
    grid[p > float(meta.get("occupied_thresh", 0.65))] = 100
    grid[p < float(meta.get("free_thresh", 0.25))] = 0

    info = {
        "resolution": float(meta["resolution"]),
        "origin": [float(v) for v in meta["origin"]],
        "height": grid.shape[0],
        "width": grid.shape[1],
    }
    return grid, info


def world_to_grid(x, y, info):
    """World metres (map frame) -> (row, col)."""
    res = info["resolution"]
    col = int((x - info["origin"][0]) / res)
    row = int((y - info["origin"][1]) / res)
    return row, col


def grid_to_world(row, col, info):
    """(row, col) -> world metres, at the cell centre."""
    res = info["resolution"]
    x = info["origin"][0] + (col + 0.5) * res
    y = info["origin"][1] + (row + 0.5) * res
    return x, y


def in_bounds(row, col, info):
    return 0 <= row < info["height"] and 0 <= col < info["width"]


def inflate(grid, info, robot_radius_m, block_unknown=True):
    """Grow obstacles by the robot radius. Returns bool array, True = blocked."""
    blocked = (grid == 100)
    if block_unknown:
        blocked = blocked | (grid == -1)

    radius_cells = int(math.ceil(robot_radius_m / info["resolution"]))
    if radius_cells <= 0:
        return blocked

    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~blocked)
    return dist <= radius_cells


# 8-connected neighbours: (d_row, d_col, step_cost)
NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def heuristic(a, b):
    """Octile distance - admissible for an 8-connected grid."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def astar(blocked, start, goal, info):
    """A* over the inflated grid. Returns list of (row, col), or None."""
    if not in_bounds(*start, info) or not in_bounds(*goal, info):
        raise ValueError("Start or goal is outside the map.")
    if blocked[start]:
        raise ValueError("Start is inside an obstacle (or too close to one).")
    if blocked[goal]:
        raise ValueError("Goal is inside an obstacle (or too close to one).")

    open_set = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        r, c = current
        for dr, dc, step in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, info) or blocked[nr, nc]:
                continue
            # don't cut diagonally through a wall corner
            if dr != 0 and dc != 0 and blocked[r + dr, c] and blocked[r, c + dc]:
                continue

            tentative = g + step
            if tentative < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative
                came_from[(nr, nc)] = current
                heapq.heappush(open_set,
                               (tentative + heuristic((nr, nc), goal), tentative, (nr, nc)))

    return None


def simplify(path):
    """Keep only the cells where direction changes - i.e. the corners.

    Raw A* gives one point per cell, far too many waypoints. The waypoint
    manager only cares about corners.
    """
    if not path or len(path) < 3:
        return path
    out = [path[0]]
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        d2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if d1 != d2:
            out.append(path[i])
    out.append(path[-1])
    return out


def plot(grid, blocked, path, waypoints, start, goal, info, title):
    fig, ax = plt.subplots(figsize=(10, 8))

    display = np.full(grid.shape, 0.5)      # grey = unknown
    display[grid == 0] = 1.0                # white = free
    display[grid == 100] = 0.0              # black = occupied
    ax.imshow(display, cmap="gray", origin="lower", vmin=0, vmax=1)

    halo = blocked & (grid != 100)
    ax.imshow(np.ma.masked_where(~halo, halo), cmap="autumn",
              origin="lower", alpha=0.25)

    if path:
        ax.plot([p[1] for p in path], [p[0] for p in path], "-",
                linewidth=2, label="A* path")
    if waypoints:
        ax.plot([p[1] for p in waypoints], [p[0] for p in waypoints], "o",
                markersize=5, label=f"waypoints ({len(waypoints)})")

    ax.plot(start[1], start[0], "s", markersize=10, label="start")
    ax.plot(goal[1], goal[0], "*", markersize=16, label="goal")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_yaml")
    ap.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--robot-radius", type=float, default=0.35)
    ap.add_argument("--allow-unknown", action="store_true")
    ap.add_argument("--info-only", action="store_true")
    args = ap.parse_args()

    grid, info = load_map(args.map_yaml)
    res, (ox, oy) = info["resolution"], info["origin"][:2]
    h, w = info["height"], info["width"]

    print("=" * 55)
    print(f"  size          : {w} x {h} cells")
    print(f"  resolution    : {res} m/cell")
    print(f"  physical size : {w*res:.2f} m x {h*res:.2f} m")
    print(f"  origin        : ({ox}, {oy})")
    print(f"  x range       : {ox:.2f} .. {ox + w*res:.2f} m")
    print(f"  y range       : {oy:.2f} .. {oy + h*res:.2f} m")
    print(f"  free          : {int((grid==0).sum())}")
    print(f"  occupied      : {int((grid==100).sum())}")
    print(f"  unknown       : {int((grid==-1).sum())}")
    print("=" * 55)

    if args.info_only:
        return

    free_rc = np.argwhere(grid == 0)
    start = (world_to_grid(*args.start, info) if args.start
             else tuple(free_rc[len(free_rc) // 4]))
    goal = (world_to_grid(*args.goal, info) if args.goal
            else tuple(free_rc[3 * len(free_rc) // 4]))

    print(f"start: {grid_to_world(*start, info)} m -> grid {start}")
    print(f"goal : {grid_to_world(*goal, info)} m -> grid {goal}")

    blocked = inflate(grid, info, args.robot_radius,
                      block_unknown=not args.allow_unknown)
    print(f"inflation: {args.robot_radius} m = "
          f"{int(math.ceil(args.robot_radius/res))} cells")

    try:
        path = astar(blocked, start, goal, info)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Try different start/goal, a smaller --robot-radius, "
              "or --allow-unknown.")
        return

    if path is None:
        print("\nNo path - start and goal are not connected through free space.")
        plot(grid, blocked, None, None, start, goal, info, "A*: NO PATH")
        return

    waypoints = simplify(path)
    length = sum(math.dist(grid_to_world(*path[i], info),
                           grid_to_world(*path[i+1], info))
                 for i in range(len(path) - 1))

    print(f"\nPATH FOUND: {len(path)} cells, {len(waypoints)} waypoints, "
          f"{length:.2f} m")
    for i, wp in enumerate(waypoints):
        x, y = grid_to_world(*wp, info)
        print(f"  W{i}: ({x:+.2f}, {y:+.2f})")

    plot(grid, blocked, path, waypoints, start, goal, info,
         f"A* - {length:.2f} m, {len(waypoints)} waypoints")


if __name__ == "__main__":
    main()
