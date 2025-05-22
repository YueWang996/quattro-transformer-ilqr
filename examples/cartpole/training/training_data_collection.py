import multiprocessing
import numpy as np
import pickle
import os
import time
import glfw
import mujoco

# --------------------------------------------------
# Import the new MPC and Transformer modules
# --------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cartpole_mpc import CartPoleMPC
from quattro_ilqr_tf.transformer_ilqr import *


def run_cartpole_sim(
    model_path: str = "../../asset/cart_pole/cartpole.xml",
    use_gui: bool = True,
    sweep_positions: np.ndarray = None,
    sweep_angles: np.ndarray = None,
    sim_steps: int = 400,
    horizon: int = 50,
    enable_mpc_sol_log: bool = False,
    mpc_log_filename: str = "ilqr_logs.pkl",
    mpc_transformer_model=None,
    flush_interval: int = None
):
    """
    Run CartPole simulation(s) using a CartPoleMPC (iLQR-based) controller.

    This function allows for:
      - sweeping multiple initial cart positions and pole angles
      - optional real-time visualization (via GLFW)
      - logging of the MPC solutions (trajectory and controls) to disk
      - periodic flushing of logs to disk if desired

    Parameters
    ----------
    model_path : str
        Path to the MuJoCo XML model file.
    use_gui : bool
        If True, opens a GLFW window for real-time visualization.
    sweep_positions : np.ndarray
        Array of cart initial positions for simulation runs.
    sweep_angles : np.ndarray
        Array of pole initial angles for simulation runs.
    sim_steps : int
        Number of simulation steps for each run.
    horizon : int
        Prediction horizon for the MPC controller.
    enable_mpc_sol_log : bool
        If True, logs the full MPC solution (predicted state and control trajectories).
    mpc_log_filename : str
        Filename for saving MPC logs.
    mpc_transformer_model : object
        Transformer model to be used (if any).
    flush_interval : int
        Number of runs after which to flush logs to disk. If None, no periodic flush is done.

    Returns
    -------
    list
        A list of MPC logs (if logging is enabled), otherwise empty or None.
    """
    # --------------------------------------------------
    # 1. Initialize the MuJoCo model and simulation data
    # --------------------------------------------------
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # --------------------------------------------------
    # 2. Initialize the MPC Controller
    # --------------------------------------------------
    mpc_controller = CartPoleMPC(
        horizon=horizon, 
        dt=0.01,
        integration_method="euler",
        transformer_model=mpc_transformer_model,
        lqr_only=False, 
        ilqr_only=True,
        ilqr_tf_only=False,
        ilqr_tf_blend=False,
        use_transformer=False  # Switch to True if you want to use the Transformer inside the iLQR
    )
    # Reference state (cart at x=0, angle=0).
    mpc_controller.x_ref = np.zeros(4)

    # --------------------------------------------------
    # 3. (Optionally) Initialize GLFW and Scene for GUI
    # --------------------------------------------------
    window, scene, context, cam, opt = None, None, None, None, None
    if use_gui:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed.")
        window = glfw.create_window(800, 600, "CartPole MuJoCo", None, None)
        if window is None:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed.")
        glfw.make_context_current(window)

        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=8000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

        # Configure camera view
        cam.lookat[:] = [0, 0, 0.5]
        cam.distance = 2.0
        cam.azimuth = 90.0
        cam.elevation = -20.0

        # Key callback to adjust cart reference position in real-time
        def key_callback(window, key, scancode, action, mods):
            if action in (glfw.PRESS, glfw.REPEAT):
                if key == glfw.KEY_LEFT:
                    mpc_controller.x_ref[0] -= 0.05
                elif key == glfw.KEY_RIGHT:
                    mpc_controller.x_ref[0] += 0.05

        glfw.set_key_callback(window, key_callback)

    # --------------------------------------------------
    # 4. Define initial conditions for the simulation sweep
    # --------------------------------------------------
    if sweep_positions is None:
        sweep_positions = [0.0]
    if sweep_angles is None:
        sweep_angles = [0.1]

    slider_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide")
    total_runs = len(sweep_positions) * len(sweep_angles)
    current_run = 0

    # If we will periodically flush logs, start with a fresh file
    if enable_mpc_sol_log and flush_interval is not None:
        try:
            open(mpc_log_filename, "wb").close()
        except Exception as e:
            print(f"Process {os.getpid()}: Warning: Could not clear log file. Error: {e}")

    # --------------------------------------------------
    # 5. Run the simulation for each (init_pos, init_angle)
    # --------------------------------------------------
    for init_pos in sweep_positions:
        for init_angle in sweep_angles:
            trial_start_time = time.time()

            # Reset simulation data
            mujoco.mj_resetData(model, data)
            data.qpos[0] = init_pos
            data.qpos[1] = init_angle

            # Perform sim_steps forward simulation
            for step_i in range(sim_steps):
                cart_x = data.qpos[0]
                pole_theta = data.qpos[1]
                cart_xdot = data.qvel[0]
                pole_thetadot = data.qvel[1]
                x_current = np.array([cart_x, cart_xdot, pole_theta, pole_thetadot])

                # Obtain control input from the MPC controller
                x_mpc, u_mpc = mpc_controller.control_step(x_current)
                data.ctrl[slider_actuator_id] = -u_mpc[0].item()

                # Step the simulation forward
                mujoco.mj_step(model, data)

                # Render scene if GUI is enabled
                if use_gui:
                    if glfw.window_should_close(window):
                        break
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), scene, context)
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        mujoco.MjrRect(10, 10, 200, 100),
                        f"Ref X: {mpc_controller.x_ref[0]:.3f}",
                        "(Left/Right Arrow to adjust)",
                        context,
                    )
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                    time.sleep(0.005)

            trial_time = time.time() - trial_start_time
            current_run += 1
            print(f"Process {os.getpid()}: [{current_run:03d}/{total_runs:03d}] "
                  f"Run finished for init_pos={init_pos:.3f}, init_angle={init_angle:.3f}. "
                  f"Running time: {trial_time:.3f} sec.")

            # (Optionally) flush logs to disk after every 'flush_interval' runs
            if enable_mpc_sol_log and flush_interval and (current_run % flush_interval == 0):
                with open(mpc_log_filename, "ab") as f:
                    pickle.dump(mpc_controller.ilqr.logs, f)
                print(f"Process {os.getpid()}: Flushed logs after {current_run:03d} runs to '{mpc_log_filename}'.")
                # Clear logs in memory
                mpc_controller.ilqr.logs.clear()

    # --------------------------------------------------
    # 6. Final cleanup and flushing any remaining logs
    # --------------------------------------------------
    if use_gui:
        glfw.terminate()

    if enable_mpc_sol_log:
        if mpc_controller.ilqr.logs:
            with open(mpc_log_filename, "ab") as f:
                pickle.dump(mpc_controller.ilqr.logs, f)
            print(f"Process {os.getpid()}: Final logs flushed to '{mpc_log_filename}'.")

    return mpc_controller.ilqr.logs


# --------------------------------------------------
# Multiprocessing task wrapper
# --------------------------------------------------
def run_task(pos_range, angles_to_try, flush_interval):
    """
    Helper function for multiprocessing. Each process handles a subset of positions
    and sweeps over all angles. Returns the path to the log file produced.

    Parameters
    ----------
    pos_range : np.ndarray
        Array of positions for this worker.
    angles_to_try : np.ndarray
        Array of angles to simulate.
    flush_interval : int
        Flush logs every N runs.

    Returns
    -------
    str
        The path to the log file created by this worker.
    """
    pos_start = pos_range[0]
    pos_end = pos_range[-1]
    log_filename = f"ilqr_logs_range_{pos_start:.3f}_{pos_end:.3f}.pkl"

    print(f"Process {os.getpid()}: Running simulation for positions from {pos_start:.3f} to {pos_end:.3f}", flush=True)
    _ = run_cartpole_sim(
        model_path="../../asset/cart_pole/cartpole.xml",
        use_gui=False,  # Recommended to disable GUI in parallel runs
        sweep_positions=pos_range,
        sweep_angles=angles_to_try,
        sim_steps=1500,
        horizon=30,
        enable_mpc_sol_log=True,
        mpc_log_filename=log_filename,
        mpc_transformer_model=None,
        flush_interval=flush_interval
    )
    print(f"Process {os.getpid()}: Finished simulation task. Log saved to {log_filename}", flush=True)
    return log_filename


# --------------------------------------------------
# Function to combine logs sequentially (like the quadrotor example)
# --------------------------------------------------
def combine_logs_sequentially(group_log_files, combined_log_filename):
    """
    Combine logs from each file into one file, reading multiple partial
    pickle dumps until EOF. Removes the individual files afterward.
    """
    print("Starting log combination...", flush=True)
    with open(combined_log_filename, "wb") as fout:
        for log_file in group_log_files:
            print(f"Combining logs from {log_file}...", flush=True)
            if not os.path.exists(log_file):
                print(f"Warning: {log_file} does not exist or was removed.", flush=True)
                continue
            with open(log_file, "rb") as fin:
                # Each flush is a separate pickle dump
                while True:
                    try:
                        partial_logs = pickle.load(fin)
                        # partial_logs should be a list of logs
                        for single_log in partial_logs:
                            pickle.dump(single_log, fout)
                    except EOFError:
                        break
            # Optionally remove the individual log file
            os.remove(log_file)
            print(f"Removed individual log file: {log_file}", flush=True)
    print(f"Combined logs saved to '{combined_log_filename}'.", flush=True)


if __name__ == "__main__":
    # --------------------------------------------------
    # 1. Define parameter ranges
    # --------------------------------------------------
    positions_to_try = np.arange(-0.5, 0.51, 0.05)
    angles_to_try = np.arange(-0.5, 0.51, 0.05)

    # --------------------------------------------------
    # 2. Configure parallelization and logging
    # --------------------------------------------------
    num_groups = 10 # Number of groups/process to split the positions into
    groups = np.array_split(positions_to_try, num_groups)
    flush_interval = 10  # Flush logs to disk every 10 runs
    combine_logs = True  # Whether to combine logs after all runs finish

    # --------------------------------------------------
    # 3. Use multiprocessing to run each group
    #    Each worker returns the path to its log file.
    # --------------------------------------------------
    # Filter out any empty groups to avoid the IndexError
    non_empty_groups = [g for g in groups if len(g) > 0]

    with multiprocessing.Pool(processes=len(non_empty_groups)) as pool:
        log_file_paths = pool.starmap(
            run_task,
            [(group, angles_to_try, flush_interval) for group in non_empty_groups]
        )

    print("Main process: All workers completed.", flush=True)

    # --------------------------------------------------
    # 4. Combine logs if desired, then remove partial logs
    # --------------------------------------------------
    if combine_logs:
        min_pos = positions_to_try.min()
        max_pos = positions_to_try.max()
        min_angle = angles_to_try.min()
        max_angle = angles_to_try.max()

        combined_log_filename = (
            f"combined_ilqr_logs_range_"
            f"{min_pos:.3f}_{max_pos:.3f}_angle_"
            f"{min_angle:.3f}_{max_angle:.3f}.pkl"
        )
        combine_logs_sequentially(log_file_paths, combined_log_filename)
        print(f"Main process: Final combined logs saved to '{combined_log_filename}'.", flush=True)