"""
Physics-based pitch trajectory simulation.
Implements ballistic motion with drag and Magnus effect.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

# Physical constants
G = 9.81  # gravity (m/s^2)
RHO = 1.225  # air density (kg/m^3)
BALL_MASS = 0.145  # kg
BALL_RADIUS = 0.0366  # m
BALL_AREA = np.pi * BALL_RADIUS ** 2
CD = 0.3  # drag coefficient
CL = 0.25  # lift coefficient (Magnus)

# Field geometry (meters)
MOUND_TO_PLATE = 18.44  # 60 ft 6 in
RELEASE_HEIGHT = 1.8  # typical release height
PLATE_HEIGHT = 0.6  # strike zone center roughly


@dataclass
class PitchParams:
    """Parameters defining a pitch."""
    velocity: float  # mph
    spin_rate: float  # rpm
    spin_axis: Tuple[float, float, float]  # unit vector
    release_angle_h: float  # horizontal angle (degrees)
    release_angle_v: float  # vertical angle (degrees)
    release_point: Tuple[float, float, float] = (0.0, RELEASE_HEIGHT, 0.0)


@dataclass 
class PitchResult:
    """Result of trajectory simulation."""
    positions: np.ndarray  # (N, 3) xyz positions
    velocities: np.ndarray  # (N, 3) xyz velocities
    times: np.ndarray  # (N,) timestamps
    plate_location: Tuple[float, float]  # (x, z) at plate
    plate_velocity: float  # mph at plate
    horizontal_break: float  # inches
    vertical_break: float  # inches (induced, gravity removed)


def mph_to_ms(mph: float) -> float:
    return mph * 0.44704


def ms_to_mph(ms: float) -> float:
    return ms / 0.44704


def rpm_to_rads(rpm: float) -> float:
    return rpm * 2 * np.pi / 60


def simulate_pitch(params: PitchParams, dt: float = 0.001) -> PitchResult:
    """
    Simulate pitch trajectory using RK4 integration.
    
    Coordinate system:
    - X: horizontal (positive = toward first base for RHP)
    - Y: height (positive = up)
    - Z: toward plate (positive = toward catcher)
    """
    v0_ms = mph_to_ms(params.velocity)
    omega = rpm_to_rads(params.spin_rate)
    
    # Initial velocity vector
    angle_h = np.radians(params.release_angle_h)
    angle_v = np.radians(params.release_angle_v)
    
    vx = v0_ms * np.sin(angle_h) * np.cos(angle_v)
    vy = v0_ms * np.sin(angle_v)
    vz = v0_ms * np.cos(angle_h) * np.cos(angle_v)
    
    # State: [x, y, z, vx, vy, vz]
    state = np.array([
        params.release_point[0],
        params.release_point[1],
        params.release_point[2],
        vx, vy, vz
    ])
    
    spin_axis = np.array(params.spin_axis)
    spin_axis = spin_axis / np.linalg.norm(spin_axis)
    
    positions = [state[:3].copy()]
    velocities = [state[3:].copy()]
    times = [0.0]
    
    t = 0.0
    while state[2] < MOUND_TO_PLATE and t < 2.0:
        state = _rk4_step(state, spin_axis, omega, dt)
        t += dt
        positions.append(state[:3].copy())
        velocities.append(state[3:].copy())
        times.append(t)
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    times = np.array(times)
    
    # Calculate metrics
    plate_loc = (positions[-1, 0], positions[-1, 1])
    plate_vel = ms_to_mph(np.linalg.norm(velocities[-1]))
    
    # Break calculation (vs no-spin trajectory)
    no_spin_end = _simulate_no_spin(params)
    h_break = (positions[-1, 0] - no_spin_end[0]) * 39.37  # to inches
    v_break = (positions[-1, 1] - no_spin_end[1]) * 39.37
    
    return PitchResult(
        positions=positions,
        velocities=velocities,
        times=times,
        plate_location=plate_loc,
        plate_velocity=plate_vel,
        horizontal_break=h_break,
        vertical_break=v_break
    )


def _compute_acceleration(pos: np.ndarray, vel: np.ndarray, 
                          spin_axis: np.ndarray, omega: float) -> np.ndarray:
    """Compute acceleration from gravity, drag, and Magnus force."""
    speed = np.linalg.norm(vel)
    if speed < 0.01:
        return np.array([0, -G, 0])
    
    vel_unit = vel / speed
    
    # Drag (opposes velocity)
    drag_mag = 0.5 * RHO * CD * BALL_AREA * speed ** 2
    drag_acc = -drag_mag / BALL_MASS * vel_unit
    
    # Magnus force (perpendicular to velocity and spin axis)
    magnus_dir = np.cross(spin_axis, vel_unit)
    magnus_mag = 0.5 * RHO * CL * BALL_AREA * speed ** 2 * (omega * BALL_RADIUS / speed)
    magnus_acc = magnus_mag / BALL_MASS * magnus_dir
    
    # Gravity
    gravity_acc = np.array([0, -G, 0])
    
    return drag_acc + magnus_acc + gravity_acc


def _rk4_step(state: np.ndarray, spin_axis: np.ndarray, 
              omega: float, dt: float) -> np.ndarray:
    """RK4 integration step."""
    def derivs(s):
        acc = _compute_acceleration(s[:3], s[3:], spin_axis, omega)
        return np.concatenate([s[3:], acc])
    
    k1 = derivs(state)
    k2 = derivs(state + 0.5 * dt * k1)
    k3 = derivs(state + 0.5 * dt * k2)
    k4 = derivs(state + dt * k3)
    
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def _simulate_no_spin(params: PitchParams) -> np.ndarray:
    """Simulate trajectory with no spin for break calculation."""
    v0_ms = mph_to_ms(params.velocity)
    angle_h = np.radians(params.release_angle_h)
    angle_v = np.radians(params.release_angle_v)
    
    vx = v0_ms * np.sin(angle_h) * np.cos(angle_v)
    vy = v0_ms * np.sin(angle_v)
    vz = v0_ms * np.cos(angle_h) * np.cos(angle_v)
    
    state = np.array([
        params.release_point[0],
        params.release_point[1],
        params.release_point[2],
        vx, vy, vz
    ])
    
    dt = 0.001
    t = 0.0
    while state[2] < MOUND_TO_PLATE and t < 2.0:
        # Only gravity and drag, no Magnus
        speed = np.linalg.norm(state[3:])
        if speed > 0.01:
            vel_unit = state[3:] / speed
            drag_mag = 0.5 * RHO * CD * BALL_AREA * speed ** 2
            drag_acc = -drag_mag / BALL_MASS * vel_unit
        else:
            drag_acc = np.zeros(3)
        
        acc = drag_acc + np.array([0, -G, 0])
        state[3:] += acc * dt
        state[:3] += state[3:] * dt
        t += dt
    
    return state[:3]
