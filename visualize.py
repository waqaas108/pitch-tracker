"""
Visualization utilities for pitch trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.physics.trajectory import PitchParams, simulate_pitch
from src.physics.camera import project_points, get_camera_presets
from src.data.pitch_types import PITCH_PROFILES, sample_pitch_params


def visualize_pitch_3d(pitch_type: str = 'FF', seed: int = 42):
    """Visualize a pitch trajectory in 3D."""
    rng = np.random.default_rng(seed)
    params_dict = sample_pitch_params(pitch_type, rng)
    params = PitchParams(**params_dict)
    result = simulate_pitch(params)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(result.positions[:, 2], result.positions[:, 0], result.positions[:, 1], 'b-', linewidth=2)
    ax1.scatter([0], [0], [params.release_point[1]], c='green', s=100, label='Release')
    ax1.scatter([result.positions[-1, 2]], [result.positions[-1, 0]], [result.positions[-1, 1]], 
                c='red', s=100, label='Plate')
    
    ax1.set_xlabel('Z (toward plate)')
    ax1.set_ylabel('X (horizontal)')
    ax1.set_zlabel('Y (height)')
    ax1.set_title(f'{PITCH_PROFILES[pitch_type].name}\n{params.velocity:.1f} mph, {params.spin_rate:.0f} rpm')
    ax1.legend()
    
    # 2D projection
    ax2 = fig.add_subplot(122)
    camera = get_camera_presets()['behind_mound_left']
    traj_2d = project_points(result.positions, camera)
    
    ax2.plot(traj_2d[:, 0], traj_2d[:, 1], 'b-', linewidth=2)
    ax2.scatter([traj_2d[0, 0]], [traj_2d[0, 1]], c='green', s=100, label='Release')
    ax2.scatter([traj_2d[-1, 0]], [traj_2d[-1, 1]], c='red', s=100, label='Plate')
    
    ax2.set_xlim(0, 1920)
    ax2.set_ylim(1080, 0)  # flip y for image coords
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_title('2D Camera View (behind mound)')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def visualize_all_pitch_types(seed: int = 42):
    """Visualize all pitch types in 2D."""
    rng = np.random.default_rng(seed)
    camera = get_camera_presets()['behind_mound_left']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (pitch_type, profile) in enumerate(PITCH_PROFILES.items()):
        if idx >= 9:
            break
            
        ax = axes[idx]
        
        # Generate multiple samples
        for _ in range(5):
            params_dict = sample_pitch_params(pitch_type, rng)
            params = PitchParams(**params_dict)
            result = simulate_pitch(params)
            traj_2d = project_points(result.positions, camera)
            
            ax.plot(traj_2d[:, 0], traj_2d[:, 1], alpha=0.5, linewidth=1)
        
        ax.set_xlim(0, 1920)
        ax.set_ylim(1080, 0)
        ax.set_title(f'{profile.name}\n~{profile.velocity_mean:.0f} mph, {profile.spin_rate_mean:.0f} rpm')
        ax.set_aspect('equal')
    
    plt.suptitle('Pitch Type Trajectories (2D Camera View)', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_break_comparison():
    """Visualize horizontal vs vertical break for all pitch types."""
    rng = np.random.default_rng(42)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(PITCH_PROFILES)))
    
    for (pitch_type, profile), color in zip(PITCH_PROFILES.items(), colors):
        h_breaks = []
        v_breaks = []
        
        for _ in range(50):
            params_dict = sample_pitch_params(pitch_type, rng)
            params = PitchParams(**params_dict)
            result = simulate_pitch(params)
            h_breaks.append(result.horizontal_break)
            v_breaks.append(result.vertical_break)
        
        ax.scatter(h_breaks, v_breaks, c=[color], label=profile.name, alpha=0.6, s=30)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Horizontal Break (inches)')
    ax.set_ylabel('Vertical Break (inches, induced)')
    ax.set_title('Pitch Movement Profile')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    output_dir = Path('./visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    
    fig1 = visualize_pitch_3d('FF')
    fig1.savefig(output_dir / 'fastball_trajectory.png', dpi=150)
    
    fig2 = visualize_pitch_3d('CU')
    fig2.savefig(output_dir / 'curveball_trajectory.png', dpi=150)
    
    fig3 = visualize_all_pitch_types()
    fig3.savefig(output_dir / 'all_pitch_types.png', dpi=150)
    
    fig4 = visualize_break_comparison()
    fig4.savefig(output_dir / 'break_comparison.png', dpi=150)
    
    print(f"Saved visualizations to {output_dir}")
    plt.show()
