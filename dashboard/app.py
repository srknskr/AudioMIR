import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torchaudio

import librosa
import soundfile as sf

from automir.audio.transforms import preprocess_audio, compute_logmel, compute_tempogram
from automir.datasets.synthetic import generate_synthetic_drum_audio
from automir.experiments.sqlite_store import ExperimentStore
from automir.inference.predictor import AutoMIRPredictor
from automir.models.factory import build_model, get_model_size_mb, count_trainable_parameters


def load_audio_bytes(audio_bytes: bytes, target_sr: int = 22050) -> tuple[torch.Tensor, int]:
    """Decode audio bytes (MP3, WAV, FLAC, etc.) into a mono PyTorch waveform tensor."""
    buf = io.BytesIO(audio_bytes)
    try:
        y, sr = librosa.load(buf, sr=target_sr, mono=True)
        waveform = torch.from_numpy(y).unsqueeze(0).float()
        return waveform, target_sr
    except Exception:
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        waveform = torch.from_numpy(data).unsqueeze(0).float()
        return waveform, sr

# Streamlit Page Config
st.set_page_config(
    page_title="AutoMIR - Multi-Objective Rhythm Intelligence",
    page_icon="🥁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look & typography
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 0.3rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_experiment_store() -> ExperimentStore:
    return ExperimentStore("results/experiments.sqlite")


def get_available_runs() -> List[Dict[str, Any]]:
    store = load_experiment_store()
    return store.get_all_runs()


def get_pareto_models_for_run(run_id: str) -> List[Dict[str, Any]]:
    run_dir = Path(f"results/{run_id}")
    retrained_path = run_dir / "retrained_pareto.json"
    pareto_path = run_dir / "pareto.json"

    if retrained_path.exists():
        with open(retrained_path, "r") as f:
            return json.load(f)
    elif pareto_path.exists():
        with open(pareto_path, "r") as f:
            return json.load(f)
    return []


# --- Sidebar ---
st.sidebar.markdown("### 🎛️ AutoMIR Controls")

# 1. Experiment Run Selector
runs = get_available_runs()
if runs:
    run_options = {f"{r['run_id']} ({r['strategy'].upper()})": r['run_id'] for r in runs}
    selected_run_label = st.sidebar.selectbox("Select Experiment Run", list(run_options.keys()))
    selected_run_id = run_options[selected_run_label]
else:
    selected_run_id = "demo_default"

# 2. Audio Input
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎵 Audio Input")
uploaded_file = st.sidebar.file_uploader("Upload Audio (WAV / MP3 / FLAC)", type=["wav", "mp3", "flac"])

use_demo_synth = False
synth_bpm = 120
synth_genre = "funk"
if uploaded_file is None:
    st.sidebar.markdown("*No file uploaded. Select synthetic demo audio:*")
    synth_genre = st.sidebar.selectbox("Synthetic Style", ["rock", "funk", "jazz", "latin"])
    synth_bpm = st.sidebar.slider("Synthetic BPM", 60, 180, 124)
    synth_meter = st.sidebar.selectbox("Synthetic Meter", ["4/4", "3/4"])
    use_demo_synth = True

# 3. Model & Preset Selector
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Pareto Presets")

pareto_candidates = get_pareto_models_for_run(selected_run_id) if selected_run_id != "demo_default" else []

preset_choice = st.sidebar.radio(
    "Pareto Preset Mode:",
    ["🏆 Best Accuracy", "⚖️ Balanced", "⚡ Fastest (Lowest Latency)", "🪶 Smallest Footprint"],
    index=1,
)

# Select candidate based on preset
selected_candidate = None
if pareto_candidates:
    if preset_choice == "🏆 Best Accuracy":
        selected_candidate = max(pareto_candidates, key=lambda c: c.get("metrics", {}).get("tempo_acc_4", 0.0))
    elif preset_choice == "⚡ Fastest (Lowest Latency)":
        selected_candidate = min(pareto_candidates, key=lambda c: c.get("metrics", {}).get("latency_ms", 999.0))
    elif preset_choice == "🪶 Smallest Footprint":
        selected_candidate = min(pareto_candidates, key=lambda c: c.get("metrics", {}).get("model_size_mb", 999.0))
    else:  # Balanced
        # Rank by normalized composite score
        def _score(c):
            m = c.get("metrics", {})
            acc = m.get("tempo_acc_4", 50.0)
            f1 = m.get("style_macro_f1", 50.0)
            lat = max(m.get("latency_ms", 10.0), 1.0)
            size = max(m.get("model_size_mb", 1.0), 0.1)
            return (acc + f1) / (lat * size)
        selected_candidate = max(pareto_candidates, key=_score)
else:
    # Default fallback demo config
    selected_candidate = {
        "candidate_id": "demo_crnn_01",
        "representation": "logmel",
        "segment_duration": 8.0,
        "n_mels": 96,
        "conv_blocks": 3,
        "base_channels": 32,
        "kernel_size": 3,
        "use_gru": True,
        "gru_hidden": 64,
        "dropout": 0.20,
        "learning_rate": 0.001,
        "batch_size": 32,
        "metrics": {
            "tempo_acc_4": 92.5,
            "style_macro_f1": 94.0,
            "latency_ms": 7.8,
            "model_size_mb": 1.25,
            "params": 320400,
        }
    }

# --- Main Dashboard ---
st.markdown("<div class='main-header'>🥁 AutoMIR: Multi-Objective Rhythm Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Automated Machine Learning for Tempo Estimation & Rhythm Understanding with Pareto Optimization</div>", unsafe_allow_html=True)

# Prepare audio
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)
    waveform, sr = load_audio_bytes(audio_bytes)
else:
    audio_np = generate_synthetic_drum_audio(
        duration=8.0,
        sample_rate=22050,
        bpm=synth_bpm,
        style=synth_genre,
        meter=synth_meter,
    )
    sr = 22050
    waveform = torch.from_numpy(audio_np).unsqueeze(0).float()
    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze().cpu().numpy(), sr, format="WAV")
    st.audio(buf.getvalue(), format="audio/wav")

# Instantiate Predictor
ckpt_path = selected_candidate.get("metrics", {}).get("checkpoint_path")
predictor = AutoMIRPredictor(
    config=selected_candidate,
    checkpoint_path=ckpt_path,
    style_classes=["rock", "funk", "jazz", "latin"],
    meter_classes=["4/4", "3/4", "6/8"],
)

with st.spinner("Running real-time inference..."):
    results = predictor.predict(audio_path_or_waveform=waveform, sample_rate=sr)

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Predicted Tempo</div>
        <div class="metric-val">{results['predicted_bpm']} <span style="font-size:1.0rem;color:#94a3b8;">BPM</span></div>
        <div class="metric-sub">Regression head: log₂(BPM)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Rhythm / Style</div>
        <div class="metric-val">{results['predicted_style'].capitalize()}</div>
        <div class="metric-sub">Confidence: {results['style_confidence']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Inference Latency</div>
        <div class="metric-val">{results['latency_ms']:.2f} <span style="font-size:1.0rem;color:#94a3b8;">ms</span></div>
        <div class="metric-sub">Batch size = 1 on {predictor.device.type.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Footprint</div>
        <div class="metric-val">{results['model_size_mb']:.2f} <span style="font-size:1.0rem;color:#94a3b8;">MB</span></div>
        <div class="metric-sub">{results['params']:,} trainable params</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for Detailed Analysis
tab_viz, tab_pareto, tab_arch = st.tabs(["📊 Audio & Feature Visualizations", "📈 Interactive Pareto Front", "⚙️ Model Architecture"])

with tab_viz:
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.subheader("Waveform")
        fig_w, ax_w = plt.subplots(figsize=(7, 2.8), facecolor="#0f172a")
        ax_w.set_facecolor("#0f172a")
        time_ax = np.arange(len(results["waveform"])) / 22050.0
        ax_w.plot(time_ax, results["waveform"], color="#38bdf8", lw=1.2)
        ax_w.set_xlabel("Time (s)", color="#94a3b8")
        ax_w.set_ylabel("Amplitude", color="#94a3b8")
        ax_w.tick_params(colors="#94a3b8")
        for spine in ax_w.spines.values():
            spine.set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig_w)

    with vcol2:
        st.subheader("Feature Representation (Log-Mel / Tempogram)")
        fig_f, ax_f = plt.subplots(figsize=(7, 2.8), facecolor="#0f172a")
        ax_f.set_facecolor("#0f172a")
        # Extract visual feature
        dur = float(selected_candidate.get("segment_duration", 8.0))
        p_wave = preprocess_audio(waveform, sample_rate=sr, target_duration=dur)
        if selected_candidate.get("representation") == "tempogram":
            feat = compute_tempogram(p_wave, sample_rate=22050).squeeze().cpu().numpy()
            img = ax_f.imshow(feat, aspect="auto", origin="lower", cmap="viridis")
            ax_f.set_ylabel("Tempo Bins", color="#94a3b8")
        else:
            feat = compute_logmel(p_wave, sample_rate=22050, n_mels=int(selected_candidate.get("n_mels", 96))).squeeze().cpu().numpy()
            img = ax_f.imshow(feat, aspect="auto", origin="lower", cmap="magma")
            ax_f.set_ylabel("Mel Bins", color="#94a3b8")
        ax_f.set_xlabel("Time Frames", color="#94a3b8")
        ax_f.tick_params(colors="#94a3b8")
        for spine in ax_f.spines.values():
            spine.set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig_f)

    # Style Probabilities
    st.subheader("Style Probabilities Distribution")
    probs_df = pd.DataFrame(list(results["style_probabilities"].items()), columns=["Genre", "Probability"])
    fig_prob = px.bar(
        probs_df,
        x="Genre",
        y="Probability",
        color="Probability",
        color_continuous_scale="Viridis",
        height=300,
    )
    fig_prob.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_prob, use_container_width=True)

with tab_pareto:
    st.subheader("Multi-Objective Pareto Trade-off Explorer")
    if selected_run_id != "demo_default":
        store = load_experiment_store()
        all_cands = store.get_candidates(selected_run_id)
    else:
        all_cands = []

    if all_cands:
        plot_data = []
        for c in all_cands:
            m = c["metrics"]
            is_pareto = c["pareto_rank"] == 0
            plot_data.append({
                "candidate_id": c["candidate_id"],
                "representation": c["config"].get("representation", "logmel"),
                "conv_blocks": c["config"].get("conv_blocks", 3),
                "use_gru": "Yes" if c["config"].get("use_gru") else "No",
                "tempo_acc_4": m.get("tempo_acc_4", 0.0),
                "style_macro_f1": m.get("style_macro_f1", 0.0),
                "latency_ms": m.get("latency_ms", 10.0),
                "model_size_mb": m.get("model_size_mb", 1.0),
                "type": "Pareto-Optimal" if is_pareto else "Evaluated Candidate",
                "size": 14 if is_pareto else 7,
            })
        df_plot = pd.DataFrame(plot_data)

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            fig_p1 = px.scatter(
                df_plot,
                x="latency_ms",
                y="tempo_acc_4",
                color="type",
                size="size",
                hover_data=["candidate_id", "representation", "conv_blocks", "use_gru", "model_size_mb"],
                color_discrete_map={"Pareto-Optimal": "#38bdf8", "Evaluated Candidate": "#64748b"},
                title="Tempo Accuracy (±4%) vs. Latency (ms)",
                labels={"latency_ms": "Inference Latency (ms)", "tempo_acc_4": "Tempo Accuracy ±4% (%)"},
            )
            fig_p1.update_layout(template="plotly_dark")
            st.plotly_chart(fig_p1, use_container_width=True)

        with pcol2:
            fig_p2 = px.scatter(
                df_plot,
                x="model_size_mb",
                y="style_macro_f1",
                color="type",
                size="size",
                hover_data=["candidate_id", "representation", "conv_blocks", "use_gru", "latency_ms"],
                color_discrete_map={"Pareto-Optimal": "#f59e0b", "Evaluated Candidate": "#64748b"},
                title="Style Macro-F1 vs. Model Size (MB)",
                labels={"model_size_mb": "Model Size (MB)", "style_macro_f1": "Style Macro-F1 (%)"},
            )
            fig_p2.update_layout(template="plotly_dark")
            st.plotly_chart(fig_p2, use_container_width=True)
    else:
        st.info("Run an AutoML search (`scripts/run_search.py`) to visualize the live interactive Pareto front.")

with tab_arch:
    st.subheader("Selected Candidate Configuration")
    st.json(selected_candidate)
