import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="FreshGuard Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Minimal CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main {background-color: #fafafa;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {
        background: white;
        border: 1px solid #eeeeee;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.03);
    }
    .small-muted {color: #666; font-size: 0.9rem;}
    .result-good {
        padding: 12px 16px;
        border-radius: 12px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-weight: 700;
    }
    .result-bad {
        padding: 12px 16px;
        border-radius: 12px;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        font-weight: 700;
    }
    div[data-testid="stTabs"] button {font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Constants
# -----------------------------
DEFAULT_CLASSES = [
    "Apple_Bad", "Apple_Good",
    "Banana_Bad", "Banana_Good",
    "Guava_Bad", "Guava_Good",
    "Lime_Bad", "Lime_Good",
    "Orange_Bad", "Orange_Good",
    "Pomegranate_Bad", "Pomegranate_Good",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -----------------------------
# Helpers
# -----------------------------
def clean_path_text(p: str) -> Path:
    return Path(p).expanduser().resolve()


def find_project_files(project_dir: Path) -> Dict[str, Optional[Path]]:
    """Find the common files created by the Colab notebook."""
    candidates = {
        "split_csv": [
            project_dir / "splits" / "freshguard_clean_group_split.csv",
            project_dir / "freshguard_clean_group_split.csv",
        ],
        "model_comparison": [
            project_dir / "results" / "model_comparison.csv",
            project_dir / "model_comparison.csv",
        ],
        "binary_metrics": [
            project_dir / "results" / "binary_monitoring_metrics.csv",
            project_dir / "binary_monitoring_metrics.csv",
        ],
        "resnet_ckpt": [
            project_dir / "checkpoints" / "resnet18_clean_best.pt",
            project_dir / "resnet18_clean_best.pt",
        ],
        "vit_ckpt": [
            project_dir / "checkpoints" / "vit_base_clean_best.pt",
            project_dir / "vit_base_clean_best.pt",
        ],
    }

    found = {}
    for key, paths in candidates.items():
        found[key] = next((p for p in paths if p.exists()), None)
    return found


def load_class_names(split_csv: Optional[Path]) -> List[str]:
    if split_csv and split_csv.exists():
        df = pd.read_csv(split_csv)
        if "label_idx" in df.columns and "class_name" in df.columns:
            return (
                df[["label_idx", "class_name"]]
                .drop_duplicates()
                .sort_values("label_idx")["class_name"]
                .tolist()
            )
        if "class_name" in df.columns:
            return sorted(df["class_name"].dropna().unique().tolist())
    return DEFAULT_CLASSES


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, checkpoint_path: str, num_classes: int):
    """Load a timm model from a saved state_dict."""
    if model_name == "ResNet-18":
        arch = "resnet18"
    elif model_name == "ViT-Base/16":
        arch = "vit_base_patch16_224"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")

    # Some checkpoints may be saved as {'model': state_dict}.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_image(model, pil_img: Image.Image, class_names: List[str]) -> Tuple[str, float, pd.DataFrame]:
    img = pil_img.convert("RGB")
    x = TFM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    pred_class = class_names[top_idx]
    pred_prob = float(probs[top_idx])

    df = pd.DataFrame({"Class": class_names, "Probability": probs})
    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    return pred_class, pred_prob, df


def class_to_quality(class_name: str) -> str:
    return "Good" if class_name.endswith("_Good") else "Bad"


def class_to_fruit(class_name: str) -> str:
    return class_name.rsplit("_", 1)[0]


def show_status_pill(quality: str):
    if quality == "Good":
        st.markdown('<div class="result-good">ACCEPT — predicted Good quality</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-bad">REJECT — predicted Bad quality</div>', unsafe_allow_html=True)


def read_csv_optional(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path and path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def metric_box(label: str, value: str, caption: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-muted">{label}</div>
            <div style="font-size:1.8rem; font-weight:800; margin-top:4px;">{value}</div>
            <div class="small-muted" style="margin-top:6px;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🍎 FreshGuard")
st.sidebar.caption("Minimal dashboard for fruit quality inspection")

project_dir_text = st.sidebar.text_input(
    "Project folder path",
    value=str(Path.cwd()),
    help="Select the folder you copied from Google Drive, for example: C:/Users/you/IMCS_FreshGuard",
)
project_dir = clean_path_text(project_dir_text)
files = find_project_files(project_dir)
class_names = load_class_names(files["split_csv"])

st.sidebar.divider()
st.sidebar.subheader("Detected files")
for label, p in files.items():
    ok = "✅" if p else "⚠️"
    st.sidebar.write(f"{ok} `{label}`")

model_choice = st.sidebar.selectbox("Model", ["ResNet-18", "ViT-Base/16"])

ckpt_path = files["resnet_ckpt"] if model_choice == "ResNet-18" else files["vit_ckpt"]
manual_ckpt = st.sidebar.text_input(
    "Checkpoint path override",
    value=str(ckpt_path) if ckpt_path else "",
    help="Use this if the checkpoint was not detected automatically.",
)
ckpt_path = clean_path_text(manual_ckpt) if manual_ckpt.strip() else None

# -----------------------------
# Header
# -----------------------------
st.title("FreshGuard Dashboard")
st.caption("Leakage-aware fruit quality inspection — simple Streamlit interface")

# -----------------------------
# Top metrics
# -----------------------------
comparison_df = read_csv_optional(files["model_comparison"])
binary_df = read_csv_optional(files["binary_metrics"])
split_df = read_csv_optional(files["split_csv"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_box("Classes", str(len(class_names)), "Fruit × Good/Bad")
with c2:
    if split_df is not None and "split" in split_df.columns:
        metric_box("Images", f"{len(split_df):,}", "From clean split CSV")
    else:
        metric_box("Images", "—", "Split CSV not found")
with c3:
    if comparison_df is not None and "Test_Accuracy_%" in comparison_df.columns:
        best_acc = comparison_df["Test_Accuracy_%"].max()
        metric_box("Best test acc.", f"{best_acc:.2f}%", "From results CSV")
    else:
        metric_box("Best test acc.", "—", "Run notebook first")
with c4:
    if binary_df is not None and "False_Accept_%" in binary_df.columns:
        fa = binary_df["False_Accept_%"].min()
        metric_box("Lowest false accept", f"{fa:.2f}%", "Bad fruit shipped")
    else:
        metric_box("False accept", "—", "Run notebook first")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab_predict, tab_metrics, tab_data, tab_about = st.tabs([
    "🔍 Inspect fruit", "📊 Results", "🧪 Dataset", "ℹ️ About"
])

with tab_predict:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Upload an image")
        uploaded = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png", "webp"])

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded image", use_container_width=True)
        else:
            pil_img = None
            st.info("Upload a fruit image to run inspection.")

    with right:
        st.subheader("Prediction")

        if pil_img is None:
            st.write("No image selected yet.")
        elif ckpt_path is None or not ckpt_path.exists():
            st.error("Checkpoint not found. Set the correct checkpoint path in the sidebar.")
        else:
            with st.spinner(f"Loading {model_choice} and predicting..."):
                try:
                    model = load_model(model_choice, str(ckpt_path), len(class_names))
                    pred_class, pred_prob, prob_df = predict_image(model, pil_img, class_names)
                    quality = class_to_quality(pred_class)
                    fruit = class_to_fruit(pred_class)

                    show_status_pill(quality)
                    st.write("")
                    cA, cB, cC = st.columns(3)
                    cA.metric("Fruit", fruit)
                    cB.metric("Quality", quality)
                    cC.metric("Confidence", f"{pred_prob*100:.2f}%")

                    st.write("Top predictions")
                    top_df = prob_df.head(5).copy()
                    top_df["Probability"] = (top_df["Probability"] * 100).map(lambda x: f"{x:.2f}%")
                    st.dataframe(top_df, use_container_width=True, hide_index=True)

                    st.bar_chart(prob_df.head(8).set_index("Class"))
                except Exception as e:
                    st.exception(e)

with tab_metrics:
    st.subheader("Model performance")

    if comparison_df is not None:
        st.write("Multiclass model comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.warning("`model_comparison.csv` was not found. Run the notebook evaluation cells first.")

    if binary_df is not None:
        st.write("Binary Good/Bad monitoring metrics")
        st.dataframe(binary_df, use_container_width=True, hide_index=True)
    else:
        st.warning("`binary_monitoring_metrics.csv` was not found. Run the notebook binary metric cells first.")

    st.markdown(
        """
        **How to read the binary metrics**

        - **False accept**: bad fruit predicted as good. This means defective fruit may be shipped.
        - **False reject**: good fruit predicted as bad. This means usable fruit may be discarded.
        """
    )

with tab_data:
    st.subheader("Clean split overview")

    if split_df is None:
        st.warning("Clean split CSV not found.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if "split" in split_df.columns:
                split_counts = split_df["split"].value_counts().rename_axis("Split").reset_index(name="Images")
                st.write("Split sizes")
                st.dataframe(split_counts, use_container_width=True, hide_index=True)
                st.bar_chart(split_counts.set_index("Split"))

        with col2:
            if "class_name" in split_df.columns:
                class_counts = split_df["class_name"].value_counts().sort_index().rename_axis("Class").reset_index(name="Images")
                st.write("Class distribution")
                st.dataframe(class_counts, use_container_width=True, hide_index=True)

        st.write("Preview")
        cols_to_show = [c for c in ["path", "class_name", "split"] if c in split_df.columns]
        st.dataframe(split_df[cols_to_show].head(20), use_container_width=True, hide_index=True)

with tab_about:
    st.subheader("About this dashboard")
    st.markdown(
        """
        This app is a lightweight interface for the FreshGuard project.

        It expects the same folder structure created by the Colab notebook:

        ```text
        IMCS_FreshGuard/
        ├── checkpoints/
        │   ├── resnet18_clean_best.pt
        │   └── vit_base_clean_best.pt
        ├── results/
        │   ├── model_comparison.csv
        │   └── binary_monitoring_metrics.csv
        └── splits/
            └── freshguard_clean_group_split.csv
        ```

        The dashboard focuses on three things:
        - quick image inspection,
        - model and monitoring metrics,
        - clean split/dataset overview.
        """
    )

    st.info("For explainability figures such as Grad-CAM/LIME, keep using the notebook unless you want a heavier dashboard version.")
