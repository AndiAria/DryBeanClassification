from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dry Bean Classification — Decision Tree vs Random Forest",
    page_icon="🫘",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / "data"
EVIDENCE_DIR = ROOT_DIR / "evidence"

MODEL_FILES = {
    "Random Forest (Tuned)": MODELS_DIR / "random_forest_tuned.joblib",
    "Decision Tree (Tuned)": MODELS_DIR / "decision_tree_tuned.joblib",
    "Random Forest (Baseline)": MODELS_DIR / "random_forest_baseline.joblib",
    "Decision Tree (Baseline)": MODELS_DIR / "decision_tree_baseline.joblib",
}


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data
def load_csv(path: Path, **kwargs):
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


@st.cache_data
def load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_data
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_dataset() -> pd.DataFrame | None:
    cleaned = OUTPUTS_DIR / "cleaned_dry_bean.csv"
    raw = DATA_DIR / "Dry_Bean.csv"
    if cleaned.exists():
        return pd.read_csv(cleaned)
    if raw.exists():
        return pd.read_csv(raw)
    return None


def show_image_if_exists(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Gambar `{path.name}` tidak ditemukan di folder `outputs/`.")


def main():
    available_models = {name: path for name, path in MODEL_FILES.items() if path.exists()}
    if not available_models:
        st.error("Folder `models/` atau file `.joblib` tidak ditemukan.")
        st.stop()

    metadata = load_json(EVIDENCE_DIR / "experiment_metadata.json")
    stats_df = load_csv(OUTPUTS_DIR / "descriptive_statistics.csv", index_col=0)
    dict_df = load_csv(ROOT_DIR / "data_dictionary.csv")
    dataset_df = load_dataset()

    with st.sidebar:
        st.header("Konfigurasi Model")
        selected_model_name = st.selectbox(
            "Model Utama untuk Prediksi",
            options=list(available_models.keys()),
            index=0,
        )
        compare_model_name = st.selectbox(
            "Model Pembanding (opsional)",
            options=["Tidak ada"] + [m for m in available_models if m != selected_model_name],
            index=1 if len(available_models) > 1 else 0,
        )

        st.divider()
        st.subheader("Ringkasan Eksperimen")
        if metadata:
            st.markdown(
                f"""
                - **Data mentah:** {metadata.get('raw_rows', 13611):,} baris
                - **Data bersih:** {metadata.get('clean_rows', 13543):,} baris
                - **Fitur:** {metadata.get('feature_count', 16)} fitur geometris
                - **Kelas:** {len(metadata.get('classes', []))} varietas
                - **Split:** 80% train / 20% test
                - **CV:** {metadata.get('cv_folds', 5)}-fold Stratified
                - **Metrik utama:** Macro F1
                """
            )
        else:
            st.markdown(
                """
                - **Model:** Decision Tree & Random Forest
                - **Fitur:** 16 fitur geometris
                - **Target:** 7 varietas Dry Bean
                """
            )

    st.title("Klasifikasi Dry Bean: Decision Tree vs Random Forest")
    st.caption(
        "Dashboard interaktif untuk prediksi varietas kacang kering (7 kelas) dan evaluasi performa model."
    )

    tab_pred, tab_eval, tab_eda = st.tabs(
        ["Prediksi Varietas", "Evaluasi & Perbandingan Model", "EDA & Ringkasan Data"]
    )

    model = load_model(available_models[selected_model_name])
    feature_names = list(model.feature_names_in_)
    feature_desc = (
        dict(zip(dict_df["column"], dict_df["description"])) if dict_df is not None else {}
    )

    with tab_pred:
        st.subheader("Prediksi Sampel Tunggal")

        preset_col1, preset_col2 = st.columns([2, 3])
        with preset_col1:
            input_mode = st.radio(
                "Sumber nilai awal fitur",
                options=["Ambil contoh dari dataset", "Nilai median dataset"],
                horizontal=True,
            )

        selected_sample_actual = None
        sample_key_suffix = "median"
        default_values = {}

        if input_mode == "Ambil contoh dari dataset" and dataset_df is not None:
            with preset_col2:
                classes = sorted(dataset_df["Class"].unique().tolist())
                chosen_class = st.selectbox("Pilih varietas contoh", options=classes)
                class_rows = dataset_df[dataset_df["Class"] == chosen_class].reset_index(drop=True)
                row_idx = st.number_input(
                    f"Indeks baris kelas {chosen_class} (0 - {len(class_rows) - 1})",
                    min_value=0,
                    max_value=len(class_rows) - 1,
                    value=0,
                    step=1,
                )
                sample_row = class_rows.iloc[int(row_idx)]
                selected_sample_actual = sample_row["Class"]
                sample_key_suffix = f"{chosen_class}_{int(row_idx)}"
                default_values = {col: float(sample_row[col]) for col in feature_names}
        elif stats_df is not None:
            default_values = {col: float(stats_df.loc[col, "50%"]) for col in feature_names}
        else:
            default_values = {col: 0.0 for col in feature_names}

        with st.form("single_prediction_form"):
            st.markdown("Sesuaikan nilai **16 fitur geometris** berikut:")
            cols = st.columns(4)
            user_inputs = {}
            for idx, feat in enumerate(feature_names):
                col = cols[idx % 4]
                val = default_values.get(feat, 0.0)
                if stats_df is not None and feat in stats_df.index:
                    f_min = float(stats_df.loc[feat, "min"]) * 0.5
                    f_max = float(stats_df.loc[feat, "max"]) * 1.5
                    val = float(np.clip(val, f_min, f_max))
                else:
                    f_min, f_max = None, None
                step = 0.0001 if abs(val) < 1.0 else (0.01 if abs(val) < 100 else 1.0)
                fmt = "%.6f" if abs(val) < 1.0 else "%.4f"
                user_inputs[feat] = col.number_input(
                    feat,
                    min_value=f_min,
                    max_value=f_max,
                    value=val,
                    step=step,
                    format=fmt,
                    help=feature_desc.get(feat),
                    key=f"input_{feat}_{input_mode}_{sample_key_suffix}",
                )

            submitted = st.form_submit_button("Prediksi Varietas", type="primary")

        if submitted or selected_sample_actual is not None:
            input_df = pd.DataFrame([user_inputs], columns=feature_names)
            pred_label = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            proba_df = pd.DataFrame(
                {"Varietas": model.classes_, "Probabilitas": proba}
            ).sort_values("Probabilitas", ascending=False)

            res_cols = st.columns(3)
            res_cols[0].metric(f"Prediksi ({selected_model_name})", pred_label)
            res_cols[1].metric("Keyakinan Tertinggi", f"{proba_df.iloc[0]['Probabilitas']:.2%}")
            if selected_sample_actual is not None and not submitted:
                res_cols[2].metric("Label Asli Sampel", selected_sample_actual)
            elif compare_model_name != "Tidak ada":
                comp_model = load_model(available_models[compare_model_name])
                comp_pred = comp_model.predict(input_df)[0]
                res_cols[2].metric(f"Pembanding ({compare_model_name})", comp_pred)

            chart_col, table_col = st.columns([3, 2])
            with chart_col:
                st.bar_chart(proba_df.set_index("Varietas")["Probabilitas"])
            with table_col:
                st.dataframe(
                    proba_df.assign(
                        Probabilitas=lambda d: d["Probabilitas"].map(lambda x: f"{x:.2%}")
                    ),
                    width="stretch",
                    hide_index=True,
                )

        st.divider()
        st.subheader("Prediksi Batch (Upload CSV)")
        uploaded_file = st.file_uploader(
            "Unggah file CSV yang memiliki 16 kolom fitur Dry Bean",
            type=["csv"],
        )
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in feature_names if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Kolom berikut tidak ditemukan pada CSV: {', '.join(missing_cols)}")
                else:
                    X_batch = batch_df[feature_names]
                    preds = model.predict(X_batch)
                    max_probs = model.predict_proba(X_batch).max(axis=1)
                    result_df = batch_df.copy()
                    result_df["Predicted_Class"] = preds
                    result_df["Confidence"] = np.round(max_probs, 4)

                    st.success(f"Berhasil memprediksi {len(result_df):,} baris data.")
                    st.dataframe(result_df.head(50), width="stretch")

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Unduh Hasil Prediksi (CSV)",
                        data=csv_bytes,
                        file_name="dry_bean_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                st.error(f"Gagal memproses file CSV: {exc}")

    with tab_eval:
        st.subheader("Performa Final pada Testing Set")
        final_metrics = load_csv(OUTPUTS_DIR / "final_test_metrics.csv")
        if final_metrics is not None:
            dt_row = final_metrics[final_metrics["model"] == "Decision Tree"].iloc[0]
            rf_row = final_metrics[final_metrics["model"] == "Random Forest"].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Random Forest Accuracy", f"{rf_row['accuracy']:.2%}")
            m2.metric("Random Forest Macro F1", f"{rf_row['f1_macro']:.2%}")
            m3.metric("Decision Tree Accuracy", f"{dt_row['accuracy']:.2%}")
            m4.metric("Decision Tree Macro F1", f"{dt_row['f1_macro']:.2%}")

            st.dataframe(final_metrics, width="stretch", hide_index=True)

        comparison_df = load_csv(OUTPUTS_DIR / "model_comparison.csv")
        if comparison_df is not None:
            st.markdown("#### Perbandingan Baseline vs Tuned")
            st.dataframe(comparison_df, width="stretch", hide_index=True)

        tuning_df = load_csv(OUTPUTS_DIR / "tuning_summary.csv")
        if tuning_df is not None:
            st.markdown("#### Ringkasan Hyperparameter Tuning (5-Fold Stratified CV)")
            st.dataframe(tuning_df, width="stretch", hide_index=True)

        st.markdown("#### Visualisasi Evaluasi")
        col_a, col_b = st.columns(2)
        with col_a:
            show_image_if_exists(
                OUTPUTS_DIR / "confusion_matrix_random_forest.png",
                "Confusion Matrix — Random Forest (Tuned)",
            )
            show_image_if_exists(
                OUTPUTS_DIR / "feature_importance_random_forest.png",
                "Feature Importance — Random Forest",
            )
        with col_b:
            show_image_if_exists(
                OUTPUTS_DIR / "confusion_matrix_decision_tree.png",
                "Confusion Matrix — Decision Tree (Tuned)",
            )
            show_image_if_exists(
                OUTPUTS_DIR / "feature_importance_decision_tree.png",
                "Feature Importance — Decision Tree",
            )

        col_c, col_d = st.columns(2)
        with col_c:
            show_image_if_exists(
                OUTPUTS_DIR / "model_comparison_macro_f1.png",
                "Perbandingan Macro F1",
            )
        with col_d:
            show_image_if_exists(
                OUTPUTS_DIR / "decision_tree_top_depth_3.png",
                "Struktur Decision Tree (Depth 3)",
            )

        rep_col1, rep_col2 = st.columns(2)
        rf_rep = load_csv(OUTPUTS_DIR / "classification_report_random_forest.csv")
        dt_rep = load_csv(OUTPUTS_DIR / "classification_report_decision_tree.csv")
        with rep_col1:
            if rf_rep is not None:
                st.markdown("#### Classification Report — Random Forest")
                st.dataframe(rf_rep, width="stretch", hide_index=True)
        with rep_col2:
            if dt_rep is not None:
                st.markdown("#### Classification Report — Decision Tree")
                st.dataframe(dt_rep, width="stretch", hide_index=True)

    with tab_eda:
        st.subheader("Ringkasan & Catatan Eksperimen")
        summary_md = load_text(OUTPUTS_DIR / "experiment_summary.md")
        notes_md = load_text(OUTPUTS_DIR / "interpretation_notes.md")
        if summary_md:
            st.markdown(summary_md)
        if notes_md:
            with st.expander("Catatan Interpretasi & Batasan Metodologi"):
                st.markdown(notes_md)

        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            show_image_if_exists(
                OUTPUTS_DIR / "class_distribution.png", "Distribusi Kelas Dry Bean"
            )
            class_dist = load_csv(OUTPUTS_DIR / "class_distribution.csv")
            if class_dist is not None:
                st.dataframe(class_dist, width="stretch", hide_index=True)
        with eda_col2:
            show_image_if_exists(
                OUTPUTS_DIR / "correlation_heatmap.png", "Heatmap Korelasi Fitur"
            )

        show_image_if_exists(
            OUTPUTS_DIR / "feature_distributions.png", "Distribusi 16 Fitur Geometris"
        )

        if stats_df is not None:
            st.markdown("#### Statistik Deskriptif Fitur")
            st.dataframe(stats_df, width="stretch")

        if dict_df is not None:
            st.markdown("#### Kamus Data (Data Dictionary)")
            st.dataframe(dict_df, width="stretch", hide_index=True)

        if dataset_df is not None:
            st.markdown("#### Pratinjau Dataset (50 Baris Pertama)")
            st.dataframe(dataset_df.head(50), width="stretch")


if __name__ == "__main__":
    main()
