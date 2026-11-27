from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dry Bean Classification — Decision Tree vs Random Forest (5-Run)",
    page_icon="🫘",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
REPEATED_MODELS_DIR = MODELS_DIR / "repeated_runs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPEATED_OUTPUTS_DIR = OUTPUTS_DIR / "repeated_runs"
DATA_DIR = ROOT_DIR / "data"
EVIDENCE_DIR = ROOT_DIR / "evidence"
REPEATED_EVIDENCE_DIR = EVIDENCE_DIR / "repeated_runs"

RUN_CONFIGS = [
    (1, 42),
    (2, 43),
    (3, 44),
    (4, 45),
    (5, 46),
]

MODEL_FILENAMES = {
    "Random Forest (Tuned)": "random_forest_tuned.joblib",
    "Decision Tree (Tuned)": "decision_tree_tuned.joblib",
    "Random Forest (Baseline)": "random_forest_baseline.joblib",
    "Decision Tree (Baseline)": "decision_tree_baseline.joblib",
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
        st.info(f"Gambar `{path.name}` tidak ditemukan.")


def get_available_runs() -> list[tuple[int, int]]:
    runs = []
    for run_num, seed in RUN_CONFIGS:
        run_dir = REPEATED_MODELS_DIR / f"run_{run_num}_seed_{seed}"
        if run_dir.exists() and any((run_dir / fn).exists() for fn in MODEL_FILENAMES.values()):
            runs.append((run_num, seed))
    return runs


def get_run_models(run_num: int, seed: int) -> dict[str, Path]:
    run_dir = REPEATED_MODELS_DIR / f"run_{run_num}_seed_{seed}"
    return {
        label: run_dir / fname
        for label, fname in MODEL_FILENAMES.items()
        if (run_dir / fname).exists()
    }


def main():
    available_runs = get_available_runs()
    if not available_runs:
        st.error("Folder `models/repeated_runs/` atau file `.joblib` belum ditemukan.")
        st.stop()

    metadata = (
        load_json(REPEATED_OUTPUTS_DIR / "experiment_metadata.json")
        or load_json(REPEATED_EVIDENCE_DIR / "experiment_metadata.json")
    )
    quality_report = load_json(EVIDENCE_DIR / "data_quality_report.json")
    stats_df = load_csv(OUTPUTS_DIR / "descriptive_statistics.csv", index_col=0)
    dict_df = load_csv(ROOT_DIR / "data_dictionary.csv")
    dataset_df = load_dataset()

    with st.sidebar:
        st.header("Konfigurasi Model")
        run_labels = {f"Run {r} (Seed {s})": (r, s) for r, s in available_runs}
        selected_run_label = st.selectbox(
            "Putaran Eksperimen (5-Run)",
            options=list(run_labels.keys()),
            index=0,
        )
        run_num, seed = run_labels[selected_run_label]
        available_models = get_run_models(run_num, seed)

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
        st.subheader("Ringkasan Eksperimen (5-Run)")
        raw_rows = (quality_report or {}).get("raw_rows", 13611)
        clean_rows = (quality_report or {}).get("clean_rows", 13543)
        exp_cfg = (metadata or {}).get("experiment_configuration", {})
        seeds = exp_cfg.get("random_states", [s for _, s in available_runs])
        best_algo = (metadata or {}).get("best_model", "Random Forest")
        st.markdown(
            f"""
            - **Data mentah:** {raw_rows:,} baris
            - **Data bersih:** {clean_rows:,} baris
            - **Fitur:** 16 fitur geometris
            - **Kelas:** 7 varietas Dry Bean
            - **Putaran:** {len(seeds)} run (`{seeds}`)
            - **Split:** 80% train / 20% test
            - **CV:** 5-fold Stratified
            - **Metrik utama:** Macro F1
            - **Model terbaik:** {best_algo}
            """
        )

    st.title("Klasifikasi Dry Bean: Decision Tree vs Random Forest")
    st.caption(
        "Dashboard interaktif eksperimen 5-run (seed 42–46) untuk prediksi varietas kacang kering dan evaluasi model."
    )

    tab_pred, tab_eval, tab_eda = st.tabs(
        ["Prediksi Varietas", "Evaluasi & Perbandingan Model (5-Run)", "EDA & Ringkasan Data"]
    )

    model = load_model(available_models[selected_model_name])
    feature_names = list(model.feature_names_in_)
    feature_desc = (
        dict(zip(dict_df["column"], dict_df["description"])) if dict_df is not None else {}
    )

    with tab_pred:
        st.subheader(f"Prediksi Sampel Tunggal — {selected_run_label}")

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
        st.subheader("Ringkasan Performa 5-Run pada Testing Set")
        perf_summary = load_csv(REPEATED_OUTPUTS_DIR / "five_run_performance_summary.csv")
        if perf_summary is not None:
            rf_tuned = perf_summary[
                (perf_summary["model"] == "Random Forest") & (perf_summary["stage"] == "Tuned")
            ]
            dt_tuned = perf_summary[
                (perf_summary["model"] == "Decision Tree") & (perf_summary["stage"] == "Tuned")
            ]
            if not rf_tuned.empty and not dt_tuned.empty:
                rf_row = rf_tuned.iloc[0]
                dt_row = dt_tuned.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Random Forest Accuracy (Mean ± Std)",
                    f"{rf_row['mean_accuracy']:.2%}",
                    f"± {rf_row['std_accuracy']:.2%}",
                )
                m2.metric(
                    "Random Forest Macro F1 (Mean ± Std)",
                    f"{rf_row['mean_f1_macro']:.2%}",
                    f"± {rf_row['std_f1_macro']:.2%}",
                )
                m3.metric(
                    "Decision Tree Accuracy (Mean ± Std)",
                    f"{dt_row['mean_accuracy']:.2%}",
                    f"± {dt_row['std_accuracy']:.2%}",
                )
                m4.metric(
                    "Decision Tree Macro F1 (Mean ± Std)",
                    f"{dt_row['mean_f1_macro']:.2%}",
                    f"± {dt_row['std_f1_macro']:.2%}",
                )
            st.dataframe(perf_summary, width="stretch", hide_index=True)

        bvt_df = load_csv(REPEATED_OUTPUTS_DIR / "five_run_baseline_vs_tuned.csv")
        if bvt_df is not None:
            st.markdown("#### Perbandingan Baseline vs Tuned (Rata-Rata 5 Run)")
            st.dataframe(bvt_df, width="stretch", hide_index=True)

        cv_df = load_csv(REPEATED_OUTPUTS_DIR / "five_run_cv_summary.csv")
        if cv_df is not None:
            st.markdown("#### Ringkasan Cross-Validation (5-Fold Stratified CV across 5 Runs)")
            st.dataframe(cv_df, width="stretch", hide_index=True)

        tuning_df = load_csv(REPEATED_OUTPUTS_DIR / "five_run_tuning_summary.csv")
        if tuning_df is not None:
            with st.expander("Detail Parameter Terbaik per Putaran (5-Run Tuning Summary)"):
                st.dataframe(tuning_df, width="stretch", hide_index=True)

        st.markdown("#### Visualisasi Rata-Rata 5-Run")
        col_a, col_b = st.columns(2)
        with col_a:
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "mean_confusion_matrix_random_forest.png",
                "Mean Normalized Confusion Matrix — Random Forest (5-Run)",
            )
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "mean_mdi_importance_random_forest.png",
                "Mean MDI Feature Importance — Random Forest (5-Run)",
            )
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "mean_permutation_importance_random_forest.png",
                "Mean Permutation Importance — Random Forest (5-Run)",
            )
        with col_b:
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "mean_confusion_matrix_decision_tree.png",
                "Mean Normalized Confusion Matrix — Decision Tree (5-Run)",
            )
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "mean_mdi_importance_decision_tree.png",
                "Mean MDI Feature Importance — Decision Tree (5-Run)",
            )
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "mean_permutation_importance_decision_tree.png",
                "Mean Permutation Importance — Decision Tree (5-Run)",
            )

        col_c, col_d = st.columns(2)
        with col_c:
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "five_run_model_comparison_macro_f1.png",
                "Perbandingan Macro F1 (5-Run)",
            )
        with col_d:
            show_image_if_exists(
                REPEATED_OUTPUTS_DIR / "five_run_cross_validation_comparison.png",
                "Perbandingan 5-Fold CV Macro F1 (5-Run)",
            )

        st.divider()
        st.markdown(f"#### Detail Evaluasi {selected_run_label}")
        run_out_dir = REPEATED_OUTPUTS_DIR / f"run_{run_num}_seed_{seed}"

        all_runs_df = load_csv(REPEATED_OUTPUTS_DIR / "five_run_final_metrics.csv")
        if all_runs_df is not None and "run" in all_runs_df.columns:
            run_metrics_df = all_runs_df[all_runs_df["run"] == run_num]
            if not run_metrics_df.empty:
                st.dataframe(run_metrics_df, width="stretch", hide_index=True)

        run_col1, run_col2 = st.columns(2)
        with run_col1:
            show_image_if_exists(
                run_out_dir / "confusion_matrix_random_forest.png",
                f"Confusion Matrix Random Forest — {selected_run_label}",
            )
        with run_col2:
            show_image_if_exists(
                run_out_dir / "confusion_matrix_decision_tree.png",
                f"Confusion Matrix Decision Tree — {selected_run_label}",
            )

        show_image_if_exists(
            run_out_dir / "decision_tree_top_depth_3.png",
            f"Struktur Decision Tree (Depth 3) — {selected_run_label}",
        )

        rep_col1, rep_col2 = st.columns(2)
        rf_rep = load_csv(run_out_dir / "classification_report_random_forest.csv")
        dt_rep = load_csv(run_out_dir / "classification_report_decision_tree.csv")
        with rep_col1:
            if rf_rep is not None:
                st.markdown(f"##### Classification Report Random Forest — {selected_run_label}")
                st.dataframe(rf_rep, width="stretch", hide_index=True)
        with rep_col2:
            if dt_rep is not None:
                st.markdown(f"##### Classification Report Decision Tree — {selected_run_label}")
                st.dataframe(dt_rep, width="stretch", hide_index=True)

    with tab_eda:
        st.subheader("Ringkasan & Catatan Eksperimen")
        summary_md = load_text(REPEATED_OUTPUTS_DIR / "experiment_summary.md")
        if summary_md:
            st.markdown(summary_md)

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
