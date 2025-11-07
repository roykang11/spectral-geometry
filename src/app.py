"""Interactive entry points for the Spectral Geometry Explorer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import geometry, solver, visualize
from .audio import synthesize_and_save

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency
    st = None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spectral Geometry Explorer CLI")
    parser.add_argument("--shape", default="circle", help="Domain shape (circle, square, rectangle, l-shape, annulus)")
    parser.add_argument("--resolution", type=int, default=80, help="Grid resolution along one axis")
    parser.add_argument("--k", type=int, default=6, help="Number of eigenmodes to compute")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo and display plots")
    parser.add_argument("--show", action="store_true", help="Display matplotlib plots")
    parser.add_argument("--save-fig", type=Path, help="Path to save a gallery of the first modes")
    parser.add_argument("--export-wav", type=Path, help="Path to export synthesized audio")
    return parser.parse_args(argv)


def run_cli(argv: list[str]) -> None:
    args = _parse_args(argv)
    params = {}
    if args.shape == "rectangle":
        params = {"width": 1.8, "height": 1.0}

    domain = geometry.create_domain(resolution=args.resolution, shape=args.shape, params=params)
    print(domain.summary())

    result = solver.solve_eigenmodes(domain, k=args.k)
    for idx, value in result.spectra():
        print(f"Mode {idx}: λ={value:.5f}")

    if args.save_fig or args.show or args.demo:
        fig = visualize.showcase_modes(result, indices=list(range(min(args.k, 6))), cols=3)
        fig.suptitle(f"Eigenmodes for {domain.name}")
        if args.save_fig:
            fig.savefig(args.save_fig, dpi=200)
            print(f"Saved figure to {args.save_fig}")
        if args.show or args.demo:
            plt.show()
        else:
            plt.close(fig)

    if args.export_wav:
        synthesize_and_save(result, args.export_wav)
        print(f"Exported audio to {args.export_wav}")

    if args.demo and not args.show:
        plt.close("all")


def run_streamlit() -> None:  # pragma: no cover - requires UI context
    st.set_page_config(page_title="Spectral Geometry Explorer", layout="wide")
    st.title("🥁 Spectral Geometry Explorer")
    st.markdown("Compute and visualize the vibration modes of 2D drums.")

    with st.sidebar:
        shape = st.selectbox("Shape", ["circle", "square", "rectangle", "l-shape", "annulus"])
        resolution = st.slider("Resolution", 40, 160, 80, step=10)
        k = st.slider("Number of modes", 3, 20, 6)
        wave_speed = st.number_input("Wave speed (m/s)", value=340.0)
        if shape == "rectangle":
            width = st.slider("Width", 0.5, 2.5, 1.8)
            height = st.slider("Height", 0.5, 2.5, 1.0)
            params = {"width": width, "height": height}
        elif shape == "annulus":
            inner = st.slider("Inner radius", 0.1, 0.9, 0.4)
            outer = st.slider("Outer radius", inner + 0.05, 1.2, 0.9)
            params = {"inner_radius": inner, "outer_radius": outer}
        else:
            params = {}

    domain = geometry.create_domain(shape=shape, resolution=resolution, params=params)
    result = solver.solve_eigenmodes(domain, k=k)

    cols = st.columns([1, 1])
    with cols[0]:
        st.subheader("Domain mask")
        fig_dom, ax_dom = plt.subplots()
        visualize.plot_domain(domain, ax=ax_dom)
        st.pyplot(fig_dom)
        plt.close(fig_dom)

    with cols[1]:
        st.subheader("Spectrum")
        fig_spec, ax_spec = plt.subplots()
        visualize.plot_spectrum(result, ax=ax_spec, wave_speed=wave_speed)
        st.pyplot(fig_spec)
        plt.close(fig_spec)

    st.subheader("Eigenmodes")
    for idx in range(k):
        fig_mode, ax_mode = plt.subplots()
        visualize.plot_mode(result, index=idx, ax=ax_mode)
        st.pyplot(fig_mode)
        plt.close(fig_mode)

    st.caption("Toggle the sidebar to explore different shapes and resolutions.")


def _running_in_streamlit() -> bool:
    if st is None:
        return False
    try:  # Streamlit 1.18+
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    if _running_in_streamlit():
        run_streamlit()
    else:
        run_cli(sys.argv[1:])


if __name__ == "__main__":
    main()
