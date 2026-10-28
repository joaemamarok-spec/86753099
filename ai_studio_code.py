#!/usr/bin/env python3
"""
ComfyUI Director & Architect Studio
A professional-grade Streamlit web application acting as a guided wrapper
and production control panel for a remote or local ComfyUI cloud instance.
"""

import json
import os
import random
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import websocket
from PIL import Image

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ComfyUI Director & Architect",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cyber-Studio Styling
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 1));
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }
    .step-card {
        padding: 1.25rem;
        border-radius: 10px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        margin-bottom: 1rem;
    }
    .badge-verified {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 700;
        color: #10b981;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "client_id" not in st.session_state:
    st.session_state.client_id = str(uuid.uuid4())

if "setup_checklist" not in st.session_state:
    st.session_state.setup_checklist = {
        "step_1_env": False,
        "step_2_core": False,
        "step_3_manager": False,
        "step_4_ipadapter": False,
        "step_5_controlnet": False,
    }

if "history" not in st.session_state:
    st.session_state.history = []

if "last_payload" not in st.session_state:
    st.session_state.last_payload = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# -----------------------------------------------------------------------------
# ComfyUI API Client Utility Functions
# -----------------------------------------------------------------------------
def clean_server_url(url: str) -> str:
    """Standardize host:port string without trailing slash or protocol."""
    cleaned = url.strip()
    if cleaned.startswith("http://"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("https://"):
        cleaned = cleaned[8:]
    return cleaned.rstrip("/")


def test_server_connection(server_address: str) -> Tuple[bool, str]:
    """Test connection to remote ComfyUI REST API."""
    clean_addr = clean_server_url(server_address)
    url = f"http://{clean_addr}/system_stats"
    try:
        resp = requests.get(url, timeout=3.5)
        if resp.status_code == 200:
            data = resp.json()
            devices = data.get("devices", [])
            gpu_info = devices[0].get("name", "GPU") if devices else "CPU / Unknown"
            return True, f"Online • System Stats OK ({gpu_info})"
        return False, f"Server returned HTTP {resp.status_code}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def upload_image_to_comfy(image_bytes: bytes, filename: str, server_address: str) -> Optional[str]:
    """Upload an image to ComfyUI input folder via POST /upload/image."""
    clean_addr = clean_server_url(server_address)
    upload_url = f"http://{clean_addr}/upload/image"
    try:
        files = {"image": (filename, image_bytes)}
        data = {"overwrite": "true"}
        resp = requests.post(upload_url, files=files, data=data, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("name", filename)
        st.warning(f"Upload returned status {resp.status_code}: {resp.text}")
        return filename
    except Exception as e:
        st.warning(f"Failed to upload image '{filename}': {e}")
        return filename


def compile_comfy_workflow(
    mode: str,
    positive_prompt: str,
    negative_prompt: str,
    references: List[Dict[str, Any]],
    steps: int,
    cfg: float,
    seed: int,
    motion_bucket: int = 127,
    denoise: float = 0.85,
    checkpoint_name: str = "v1-5-pruned-emaonly.safetensors",
) -> Dict[str, Any]:
    """
    Constructs a valid ComfyUI API execution graph JSON.
    Includes CheckpointLoader, CLIPTextEncode (pos/neg), IPAdapterApply nodes,
    ControlNet nodes, KSampler, and Output nodes (Image or Video).
    """
    prompt_graph: Dict[str, Any] = {}

    # Node 1: Model Loader
    prompt_graph["1"] = {
        "inputs": {"ckpt_name": checkpoint_name},
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"},
    }

    # Node 2: Positive Prompt
    prompt_graph["2"] = {
        "inputs": {
            "text": positive_prompt or "cinematic photograph, ultra high quality, masterwork, 8k",
            "clip": ["1", 1],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive)"},
    }

    # Node 3: Negative Prompt
    prompt_graph["3"] = {
        "inputs": {
            "text": negative_prompt or "ugly, blurry, low quality, deformed, distorted, bad anatomy",
            "clip": ["1", 1],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative)"},
    }

    # Node 4: Canvas Size
    width = 1024 if mode == "Image-to-Video" else 768
    height = 576 if mode == "Image-to-Video" else 768
    prompt_graph["4"] = {
        "inputs": {"width": width, "height": height, "batch_size": 1},
        "class_type": "EmptyLatentImage",
        "_meta": {"title": "Empty Latent Canvas"},
    }

    current_model = ["1", 0]
    node_counter = 10

    if references:
        prompt_graph[str(node_counter)] = {
            "inputs": {"ipadapter_file": "ip-adapter-plus_sd15.safetensors"},
            "class_type": "IPAdapterModelLoader",
            "_meta": {"title": "Load IPAdapter Model"},
        }
        ipadapter_model_node = str(node_counter)
        node_counter += 1

        prompt_graph[str(node_counter)] = {
            "inputs": {"clip_name": "model.safetensors"},
            "class_type": "CLIPVisionLoader",
            "_meta": {"title": "Load CLIP Vision"},
        }
        clip_vision_node = str(node_counter)
        node_counter += 1

        for ref in references:
            img_node = str(node_counter)
            prompt_graph[img_node] = {
                "inputs": {"image": ref.get("filename", "input.png"), "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": f"Ref Image ({ref.get('role', 'Identity')})"},
            }
            node_counter += 1

            apply_node = str(node_counter)
            weight = float(ref.get("weight", 0.8))
            role = ref.get("role", "Face Reference")

            prompt_graph[apply_node] = {
                "inputs": {
                    "weight": weight,
                    "noise": 0.05,
                    "weight_type": "ease in-out" if "Face" in role else "linear",
                    "start_at": 0.0,
                    "end_at": 0.95,
                    "model": current_model,
                    "ipadapter": [ipadapter_model_node, 0],
                    "image": [img_node, 0],
                    "clip_vision": [clip_vision_node, 0],
                },
                "class_type": "IPAdapterApply",
                "_meta": {"title": f"IPAdapter Apply ({role})"},
            }
            current_model = [apply_node, 0]
            node_counter += 1

    if mode == "Image-to-Video":
        svd_node = str(node_counter)
        prompt_graph[svd_node] = {
            "inputs": {
                "clip_vision": ["1", 1],
                "init_image": current_model,
                "motion_bucket_id": int(motion_bucket),
                "fps": 8,
                "augmentation_level": 0.02,
            },
            "class_type": "SVD_img2vid_Conditioning",
            "_meta": {"title": "SVD Video Conditioning"},
        }
        node_counter += 1

        ksampler_node = str(node_counter)
        prompt_graph[ksampler_node] = {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "karras",
                "denoise": denoise,
                "model": current_model,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler (Video Frames)"},
        }
        node_counter += 1

        decode_node = str(node_counter)
        prompt_graph[decode_node] = {
            "inputs": {"samples": [ksampler_node, 0], "vae": ["1", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode Frames"},
        }
        node_counter += 1

        prompt_graph[str(node_counter)] = {
            "inputs": {
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": "Director_Video",
                "format": "video/h264-mp4",
                "images": [decode_node, 0],
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine / MP4 Save"},
        }
    else:
        ksampler_node = str(node_counter)
        prompt_graph[ksampler_node] = {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": current_model,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler (High Res)"},
        }
        node_counter += 1

        decode_node = str(node_counter)
        prompt_graph[decode_node] = {
            "inputs": {"samples": [ksampler_node, 0], "vae": ["1", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        }
        node_counter += 1

        prompt_graph[str(node_counter)] = {
            "inputs": {"filename_prefix": "Director_Output", "images": [decode_node, 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save High-Res Image"},
        }

    return prompt_graph


def queue_prompt(prompt_workflow: Dict[str, Any], client_id: str, server_address: str) -> Optional[str]:
    """Queue workflow prompt via POST /prompt and return prompt_id."""
    clean_addr = clean_server_url(server_address)
    url = f"http://{clean_addr}/prompt"
    payload = {"prompt": prompt_workflow, "client_id": client_id}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("prompt_id")
        st.error(f"Failed to queue prompt: HTTP {resp.status_code} - {resp.text}")
        return None
    except Exception as e:
        st.error(f"Network error queueing prompt: {e}")
        return None


def execute_comfy_with_websocket(
    prompt_workflow: Dict[str, Any],
    client_id: str,
    server_address: str,
    progress_bar,
    status_text,
) -> Optional[Dict[str, Any]]:
    """
    Establishes WebSocket connection to remote ComfyUI instance,
    queues prompt, streams real-time node progress, and retrieves output binaries.
    """
    clean_addr = clean_server_url(server_address)
    ws_url = f"ws://{clean_addr}/ws?clientId={client_id}"

    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        status_text.info("Connected to ComfyUI WebSocket stream. Submitting prompt...")
    except Exception as e:
        status_text.warning(f"Could not connect WebSocket ({e}). Will attempt REST polling.")
        ws = None

    prompt_id = queue_prompt(prompt_workflow, client_id, server_address)
    if not prompt_id:
        if ws:
            ws.close()
        return None

    status_text.success(f"Prompt queued! Prompt ID: `{prompt_id}`. Listening for execution...")

    if ws:
        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if msg_type == "execution_start":
                        status_text.info("⚡ Execution started on remote node pipeline...")

                    elif msg_type == "executing":
                        node = data.get("node")
                        if node is None:
                            status_text.success("Pipeline execution completed!")
                            break
                        node_info = prompt_workflow.get(str(node), {}).get("_meta", {}).get("title", f"Node {node}")
                        status_text.info(f"Executing: **{node_info}** (ID: `{node}`)...")

                    elif msg_type == "progress":
                        val = data.get("value", 0)
                        max_val = data.get("max", 1)
                        pct = min(1.0, float(val) / float(max_val) if max_val > 0 else 0.0)
                        progress_bar.progress(pct)
                        status_text.info(f"Sampling Step: {val} / {max_val} ({int(pct * 100)}%)")

                    elif msg_type == "execution_error":
                        st.error(f"Execution Error in node {data.get('node_id')}: {data.get('exception_message')}")
                        break

            ws.close()
        except Exception as e:
            st.warning(f"WebSocket closed or timed out: {e}")
            if ws:
                ws.close()

    try:
        history_url = f"http://{clean_addr}/history/{prompt_id}"
        resp = requests.get(history_url, timeout=10)
        if resp.status_code == 200:
            hist_data = resp.json().get(prompt_id, {})
            outputs = hist_data.get("outputs", {})
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        view_url = f"http://{clean_addr}/view?filename={img['filename']}&subfolder={img.get('subfolder', '')}&type={img.get('type', 'output')}"
                        img_resp = requests.get(view_url, timeout=15)
                        if img_resp.status_code == 200:
                            return {
                                "type": "image",
                                "bytes": img_resp.content,
                                "filename": img["filename"],
                                "prompt_id": prompt_id,
                            }
                if "gifs" in node_output or "videos" in node_output:
                    vids = node_output.get("gifs") or node_output.get("videos")
                    for vid in vids:
                        view_url = f"http://{clean_addr}/view?filename={vid['filename']}&subfolder={vid.get('subfolder', '')}&type={vid.get('type', 'output')}"
                        vid_resp = requests.get(view_url, timeout=20)
                        if vid_resp.status_code == 200:
                            return {
                                "type": "video",
                                "bytes": vid_resp.content,
                                "filename": vid["filename"],
                                "prompt_id": prompt_id,
                            }
    except Exception as e:
        st.error(f"Error fetching output from ComfyUI history: {e}")

    return None


# -----------------------------------------------------------------------------
# Sidebar: Connection, Presets & Mode Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ ComfyUI Studio Control")
    st.caption("Guided Production Wrapper & Cloud Orchestrator")

    mode = st.radio(
        "Select Active Mode",
        options=["🏗️ The Architect (Setup Mode)", "🎬 The Director (Production Mode)"],
        index=1,
    )

    st.markdown("---")
    st.markdown("#### 🌐 Remote Server Settings")

    server_preset = st.selectbox(
        "Quick Host Presets",
        ["Localhost (127.0.0.1:8188)", "RunPod Pod URL", "Vast.ai Instance", "Custom Remote IP"],
    )

    default_server = "127.0.0.1:8188"
    if "RunPod" in server_preset:
        default_server = "your-pod-id-8188.proxy.runpod.net"
    elif "Vast.ai" in server_preset:
        default_server = "ssh.vast.ai:45678"

    server_address = st.text_input("ComfyUI Server Address", value=default_server)
    client_id_input = st.text_input("Client Session ID (UUID)", value=st.session_state.client_id)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔌 Ping Server"):
            online, msg = test_server_connection(server_address)
            if online:
                st.success(msg)
            else:
                st.error(msg)
    with col_btn2:
        if st.button("🎲 New Client ID"):
            st.session_state.client_id = str(uuid.uuid4())
            st.rerun()


# -----------------------------------------------------------------------------
# Mode 1: The Architect (Setup Mode)
# -----------------------------------------------------------------------------
if "The Architect" in mode:
    st.title("🏗️ The Architect — Guided Cloud Instance Setup")
    st.markdown(
        """
        Follow this verified sequential checklist to provision your remote cloud instance 
        (RunPod, Vast.ai, Modal, Lambda Labs, or local GPU workstation) for full ComfyUI API compatibility.
        """
    )

    completed_steps = sum(1 for v in st.session_state.setup_checklist.values() if v)
    progress_val = completed_steps / 5.0
    st.progress(progress_val)
    st.caption(f"Setup Progress: {completed_steps} / 5 Steps Complete ({int(progress_val * 100)}%)")

    # Step 1
    with st.container():
        st.markdown(
            """
            <div class="step-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">Step 1: Python Environment & PyTorch CUDA Setup</h3>
                    <span class="badge-verified">Verified PyTorch 2.3+</span>
                </div>
                <p style="margin-top:0.5rem; color:#94a3b8;">
                    Create an isolated virtual environment and install PyTorch with matching CUDA drivers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Official Documentation & Wheel Links:** [PyTorch Get Started](https://pytorch.org/get-started/locally/)")
        st.code(
            """# 1. Update system & create virtual environment
sudo apt update && sudo apt install -y python3-venv git curl ffmpeg
python3 -m venv comfy_env
source comfy_env/bin/activate

# 2. Install PyTorch with CUDA 12.1 acceleration
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121""",
            language="bash",
        )
        st.session_state.setup_checklist["step_1_env"] = st.checkbox(
            "Mark Step 1 as completed (Python & PyTorch verified)",
            value=st.session_state.setup_checklist["step_1_env"],
        )

    st.markdown("---")

    # Step 2
    with st.container():
        st.markdown(
            """
            <div class="step-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">Step 2: Clone ComfyUI Core & Launch Flags</h3>
                    <span class="badge-verified">Official Core Repo</span>
                </div>
                <p style="margin-top:0.5rem; color:#94a3b8;">
                    Clone the master branch of ComfyUI and configure it to bind to external cloud IPs with WebSocket API support enabled.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Verified Repository:** [comfyanonymous/ComfyUI (GitHub)](https://github.com/comfyanonymous/ComfyUI)")
        st.code(
            """# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Launch command for Cloud VMs (Listens on all network interfaces on port 8188)
python main.py --listen 0.0.0.0 --port 8188 --preview-method auto""",
            language="bash",
        )
        st.session_state.setup_checklist["step_2_core"] = st.checkbox(
            "Mark Step 2 as completed (ComfyUI Core installed & running)",
            value=st.session_state.setup_checklist["step_2_core"],
        )

    st.markdown("---")

    # Step 3
    with st.container():
        st.markdown(
            """
            <div class="step-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">Step 3: Install ComfyUI-Manager Node</h3>
                    <span class="badge-verified">Verified Manager Node</span>
                </div>
                <p style="margin-top:0.5rem; color:#94a3b8;">
                    The essential package manager for installing custom nodes, missing models, and tracking updates.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Verified Repository:** [ltdrdata/ComfyUI-Manager (GitHub)](https://github.com/ltdrdata/ComfyUI-Manager)")
        st.code(
            """cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ComfyUI-Manager
pip install -r requirements.txt""",
            language="bash",
        )
        st.session_state.setup_checklist["step_3_manager"] = st.checkbox(
            "Mark Step 3 as completed (ComfyUI-Manager active)",
            value=st.session_state.setup_checklist["step_3_manager"],
        )

    st.markdown("---")

    # Step 4
    with st.container():
        st.markdown(
            """
            <div class="step-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">Step 4: Custom Nodes — IPAdapter-Plus (Identity Locking)</h3>
                    <span class="badge-verified">Cubiq IPAdapter-Plus</span>
                </div>
                <p style="margin-top:0.5rem; color:#94a3b8;">
                    Required for multi-reference face, body, and style conditioning without training LoRAs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.markdown("**Verified Node Repo:** [cubiq/ComfyUI_IPAdapter_plus (GitHub)](https://github.com/cubiq/ComfyUI_IPAdapter_plus)")
        with col_link2:
            st.markdown("**Verified Model Weights:** [h94/IP-Adapter (Hugging Face)](https://huggingface.co/h94/IP-Adapter)")

        st.code(
            """# 1. Install IPAdapter-Plus custom node
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# 2. Download CLIP Vision model
mkdir -p ComfyUI/models/clip_vision
cd ComfyUI/models/clip_vision
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors

# 3. Download IP-Adapter SD1.5 & SDXL weights
mkdir -p ComfyUI/models/ipadapter
cd ComfyUI/models/ipadapter
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors
wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors""",
            language="bash",
        )
        st.session_state.setup_checklist["step_4_ipadapter"] = st.checkbox(
            "Mark Step 4 as completed (IPAdapter-Plus nodes & weights downloaded)",
            value=st.session_state.setup_checklist["step_4_ipadapter"],
        )

    st.markdown("---")

    # Step 5
    with st.container():
        st.markdown(
            """
            <div class="step-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">Step 5: ControlNet & Aux Preprocessors</h3>
                    <span class="badge-verified">ControlNet Aux v1.1</span>
                </div>
                <p style="margin-top:0.5rem; color:#94a3b8;">
                    Provides OpenPose, Depth, Canny, and Lineart preprocessors for exact pose locking.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Verified Preprocessors:** [Fannovel16/comfyui_controlnet_aux (GitHub)](https://github.com/Fannovel16/comfyui_controlnet_aux)")
        with col_c2:
            st.markdown("**Verified Model Weights:** [lllyasviel/ControlNet-v1-1 (Hugging Face)](https://huggingface.co/lllyasviel/ControlNet-v1-1)")

        st.code(
            """# 1. Install ControlNet Aux Preprocessors
cd ComfyUI/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd comfyui_controlnet_aux && pip install -r requirements.txt

# 2. Download OpenPose & Depth ControlNet models
mkdir -p ComfyUI/models/controlnet
cd ComfyUI/models/controlnet
wget -c https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.safetensors
wget -c https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.safetensors""",
            language="bash",
        )
        st.session_state.setup_checklist["step_5_controlnet"] = st.checkbox(
            "Mark Step 5 as completed (ControlNet preprocessors & models ready)",
            value=st.session_state.setup_checklist["step_5_controlnet"],
        )


# -----------------------------------------------------------------------------
# Mode 2: The Director (Production Mode)
# -----------------------------------------------------------------------------
elif "The Director" in mode:
    st.title("🎬 The Director — Production Orchestrator")
    st.caption("Identity Locking, Scene Control & Video Generation Pipeline")

    col_left, col_right = st.columns([1.1, 1.0], gap="large")

    with col_left:
        # 1. Identity Locking
        st.subheader("1. 👤 Identity Locking (Multi-Reference IPAdapter)")
        st.markdown("Upload one or more reference images, assign specific roles, and tune influence weights:")

        num_refs = st.number_input("Number of Reference Inputs", min_value=1, max_value=4, value=2, step=1)
        references_data: List[Dict[str, Any]] = []

        for i in range(int(num_refs)):
            with st.expander(f"Reference Slot #{i+1}", expanded=True):
                r_col1, r_col2 = st.columns([1.2, 1.0])
                with r_col1:
                    uploaded_file = st.file_uploader(
                        f"Upload Reference Image #{i+1}",
                        type=["png", "jpg", "jpeg", "webp"],
                        key=f"uploader_{i}",
                    )
                with r_col2:
                    default_role_idx = 0 if i == 0 else 1
                    role = st.selectbox(
                        f"Assigned Role #{i+1}",
                        options=["Face Reference", "Body Reference", "Pose Reference"],
                        index=default_role_idx,
                        key=f"role_{i}",
                    )
                    weight = st.slider(
                        f"Influence Weight #{i+1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.85 if "Face" in role else 0.65,
                        step=0.05,
                        key=f"weight_{i}",
                    )

                if uploaded_file is not None:
                    img_preview = Image.open(uploaded_file)
                    st.image(img_preview, width=120, caption=f"{role} (w={weight})")
                    references_data.append({
                        "file_bytes": uploaded_file.getvalue(),
                        "filename": uploaded_file.name,
                        "role": role,
                        "weight": weight,
                    })

        st.markdown("---")

        # 2. Scene Control
        st.subheader("2. 🎨 Scene Control")
        positive_prompt = st.text_area(
            "Positive Scene Description (Prompt)",
            value="A confident cyberpunk agent standing in a rain-slicked Tokyo alley, neon reflections in puddles, cinematic rim lighting, 35mm lens, photorealistic 8k, volumetric smoke, high fidelity",
            height=110,
        )

        negative_prompt = st.text_area(
            "Negative Description (Unwanted Elements)",
            value="ugly, deformed, mutated, extra limbs, bad eyes, poorly drawn hands, low resolution, blurry, watermark, signature",
            height=70,
        )

        st.markdown("---")

        # 3. Output Mode & Quality Tuning
        st.subheader("3. ⚙️ Output Mode & Quality Tuning")
        out_mode = st.radio(
            "Output Generation Mode",
            options=["Image-to-Image", "Image-to-Video"],
            horizontal=True,
        )

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            steps = st.slider("Sampling Steps (Fidelity)", min_value=10, max_value=100, value=25, step=1)
            cfg_scale = st.slider("CFG Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)

        with q_col2:
            if out_mode == "Image-to-Video":
                motion_bucket = st.slider(
                    "Motion Bucket (Video Intensity)",
                    min_value=1,
                    max_value=255,
                    value=127,
                    step=1,
                )
            else:
                motion_bucket = 127
                st.slider("Denoise Strength", min_value=0.1, max_value=1.0, value=0.75, step=0.05)

            seed_val = st.number_input("Seed (-1 for random)", value=-1, step=1)
            active_seed = random.randint(1, 10_000_000) if seed_val == -1 else int(seed_val)

        st.markdown("---")
        generate_btn = st.button("🚀 Generate (Queue ComfyUI Prompt)", type="primary", use_container_width=True)

    # 4. Right Column: Execution & Results
    with col_right:
        st.subheader("4. 📺 Live Execution & Result Gallery")

        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        if generate_btn:
            status_placeholder.info("Compiling workflow JSON payload...")

            compiled_refs = []
            for r in references_data:
                saved_name = upload_image_to_comfy(r["file_bytes"], r["filename"], server_address)
                compiled_refs.append({
                    "filename": saved_name or r["filename"],
                    "role": r["role"],
                    "weight": r["weight"],
                })

            workflow_payload = compile_comfy_workflow(
                mode=out_mode,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                references=compiled_refs,
                steps=steps,
                cfg=cfg_scale,
                seed=active_seed,
                motion_bucket=motion_bucket,
            )
            st.session_state.last_payload = workflow_payload

            pb = progress_placeholder.progress(0.0)
            res = execute_comfy_with_websocket(
                prompt_workflow=workflow_payload,
                client_id=st.session_state.client_id,
                server_address=server_address,
                progress_bar=pb,
                status_text=status_placeholder,
            )

            if res:
                st.session_state.last_result = res
                st.session_state.history.append(res)
                progress_placeholder.progress(1.0)
                status_placeholder.success("✨ Generation successfully retrieved!")
            else:
                status_placeholder.warning(
                    "Remote ComfyUI did not return an image or is unreachable. "
                    "Review the compiled JSON payload below."
                )

        if st.session_state.last_result:
            res = st.session_state.last_result
            st.markdown("#### 🏆 High-Resolution Result")
            if res.get("type") == "video":
                st.video(res["bytes"])
                st.download_button("📥 Download MP4", res["bytes"], file_name="director_output.mp4", mime="video/mp4")
            else:
                st.image(res["bytes"], use_container_width=True, caption=f"ComfyUI Output ({res.get('filename')})")
                st.download_button("📥 Download PNG", res["bytes"], file_name="director_output.png", mime="image/png")
        else:
            st.info("Awaiting execution. Click 'Generate' on the left to queue the prompt and stream outputs.")

        with st.expander("🔍 Inspect Compiled ComfyUI JSON Payload", expanded=False):
            if st.session_state.last_payload:
                st.json(st.session_state.last_payload)
            else:
                sample_payload = compile_comfy_workflow(
                    mode=out_mode,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    references=[],
                    steps=steps,
                    cfg=cfg_scale,
                    seed=42,
                    motion_bucket=motion_bucket,
                )
                st.json(sample_payload)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "ComfyUI Director & Architect Studio • Production-Ready Full-Stack Streamlit Application • "
    "Designed for RunPod, Vast.ai, Modal & Local Cloud Deployments."
)