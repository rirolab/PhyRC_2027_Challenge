import h5py
import numpy as np
from pathlib import Path

HDF5 = "research_data/perception_validation/machine_01/policy.hdf5"
OUT = Path("research_output/grasp_verification_v1")
OUT.mkdir(parents=True, exist_ok=True)

with h5py.File(HDF5, "r") as f:
    obs = f["data/demo_0/obs"]

    time = np.asarray(obs["simulation_time_s"])[:, 0]
    command = np.asarray(obs["gripper_close_command"]).reshape(-1, 2)
    distance = np.asarray(obs["fingertip_origin_distance"]).reshape(-1, 2)
    positions = np.asarray(obs["fingertip_position_base"]).reshape(-1, 2, 2, 3)

    geometric_distance = np.linalg.norm(
        positions[:, :, 0] - positions[:, :, 1],
        axis=-1
    )

    error = np.abs(distance - geometric_distance)

    print("=== GEOMETRIC CONSISTENCY ===")
    print("Max absolute error (m):", float(error.max()))
    print("Mean absolute error (m):", float(error.mean()))

    for robot in range(2):
        print(f"\n=== ROBOT {robot} ===")

        c = command[:, robot].astype(bool)
        d = distance[:, robot]

        print("Close-command fraction:", round(float(c.mean()), 4))
        print("Distance min/max (m):", float(d.min()), float(d.max()))
        print("Distance median (m):", float(np.median(d)))

        changes = np.flatnonzero(c[1:] != c[:-1]) + 1
        print("Command transitions:", len(changes))

        for idx in changes:
            action = "CLOSE" if c[idx] else "OPEN"
            end = min(idx + 10, len(d) - 1)
            delta = d[end] - d[idx]

            print(
                f"frame={idx:4d} "
                f"t={time[idx]:7.3f}s "
                f"{action:5s} "
                f"d_before={d[idx-1]:.5f} "
                f"d_now={d[idx]:.5f} "
                f"d_after={d[end]:.5f} "
                f"delta_10frames={delta:+.5f}"
            )

    print("\n=== SELECTED FRAMES ===")
    for frame in (1700, 1730, 1760, 2350, 2450):
        print(
            frame,
            "t=", round(float(time[frame]), 3),
            "close=", command[frame].astype(int).tolist(),
            "distance_m=", distance[frame].round(5).tolist()
        )

    np.savez(
        OUT / "gripper_signals.npz",
        simulation_time_s=time,
        close_command=command,
        fingertip_distance_m=distance,
        geometric_distance_m=geometric_distance,
    )

print("\nSaved:", OUT / "gripper_signals.npz")
