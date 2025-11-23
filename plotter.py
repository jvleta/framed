import numpy as np
import matplotlib.pyplot as plt


def plot_frame(nodes, elements, displacements=None, scale_factor=1.0):
    """Plot original and optionally deformed frame configurations."""
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)

    plt.figure(figsize=(12, 8))

    for idx, element in enumerate(elements):
        node1_idx, node2_idx = element
        x_coords = [nodes[node1_idx][0], nodes[node2_idx][0]]
        y_coords = [nodes[node1_idx][1], nodes[node2_idx][1]]
        plt.plot(
            x_coords,
            y_coords,
            "b-",
            linewidth=3,
            alpha=0.7,
            label="Original" if idx == 0 else "",
        )

    plt.scatter(
        nodes[:, 0], nodes[:, 1], c="blue", s=50, zorder=5, label="Original nodes"
    )

    if displacements is not None:
        displacements = np.asarray(displacements, dtype=float)
        u_displacements = displacements.reshape(-1, 3)[:, :2]
        deformed_nodes = nodes + u_displacements * scale_factor

        for idx, element in enumerate(elements):
            node1_idx, node2_idx = element
            x_coords = [deformed_nodes[node1_idx][0], deformed_nodes[node2_idx][0]]
            y_coords = [deformed_nodes[node1_idx][1], deformed_nodes[node2_idx][1]]
            plt.plot(
                x_coords,
                y_coords,
                "r--",
                linewidth=3,
                alpha=0.7,
                label="Deformed" if idx == 0 else "",
            )

        plt.scatter(
            deformed_nodes[:, 0],
            deformed_nodes[:, 1],
            c="red",
            s=50,
            zorder=5,
            label="Deformed nodes",
        )

        for i, node in enumerate(nodes):
            dx, dy = (u_displacements[i] * scale_factor).tolist()
            if abs(dx) > 1e-8 or abs(dy) > 1e-8:
                plt.arrow(
                    node[0],
                    node[1],
                    dx,
                    dy,
                    head_width=0.03,
                    head_length=0.03,
                    fc="green",
                    ec="green",
                    alpha=0.7,
                )

    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.5, 6)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(
        f"Frame Structure{' - Deformed vs Original' if displacements is not None else ''}"
    )
    plt.xlabel("X-axis (m)")
    plt.ylabel("Y-axis (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if displacements is not None and scale_factor != 1.0:
        plt.text(
            0.02,
            0.98,
            f"Deformation scale: {scale_factor}x",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def plot_load_displacement(load_cases, max_displacements):
    """Plot load vs displacement and linear fit for quick visual verification."""
    loads = np.asarray(load_cases, dtype=float)
    disps_m = np.asarray(max_displacements, dtype=float)

    if loads.size == 0 or disps_m.size == 0:
        return

    disps_mm = disps_m * 1000.0

    plt.figure(figsize=(10, 6))
    plt.plot(loads, disps_mm, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Applied Load (N)")
    plt.ylabel("Maximum Displacement (mm)")
    plt.title("Load vs Maximum Displacement (Linear Elastic Analysis)")
    plt.grid(True, alpha=0.3)

    has_load = loads[-1] != 0
    slope_m_per_n = disps_m[-1] / loads[-1] if has_load else 0.0
    slope_mm_per_n = slope_m_per_n * 1000
    linear_fit_mm = slope_m_per_n * loads * 1000 if has_load else np.zeros_like(loads)
    plt.plot(
        loads,
        linear_fit_mm,
        "r--",
        alpha=0.7,
        label=f"Linear fit (slope={slope_mm_per_n:.3f} mm/N)" if has_load else "Linear fit",
    )
    plt.legend()

    plt.tight_layout()
    plt.show()
