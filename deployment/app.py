"""Streamlit entry point for the final demonstration."""


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")
    st.title("Industrial Surface Anomaly Detection")
    st.caption("Upload image → model inference → score → heatmap → mask → overlay")
    st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])
    st.selectbox("Category", ["wood", "metal_nut", "capsule", "cable"])
    st.selectbox("Model", ["Autoencoder", "PaDiM", "PatchCore"])
    st.info("Inference adapter will be connected after the model interface is frozen.")


if __name__ == "__main__":
    main()

