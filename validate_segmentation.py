import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from csf_flow import compute_flow_and_stroke_volume

# Parameters
DATA_ROOT = Path("data/test")
PRED_MASK_DIR = Path("outputs/preds_single")
META_CSV = Path("merged_metadata_file.csv")
FIG_DIR = Path("outputs/validation_figures")
OUT_CSV = Path("outputs/stroke_volume_validation.csv")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata table
meta_df = pd.read_csv(META_CSV)

results = []

for pred_file in sorted(PRED_MASK_DIR.glob("*_pred.npy")):
    subj_id = pred_file.stem.replace("_pred", "")
    subj_dir = DATA_ROOT / subj_id

    try:
        # Load data
        phase = np.load(subj_dir / "phase.npy") * 2 - 1  # [0,1] -> [-1,1]
        mask_pred = np.load(pred_file)
        mask_gt = np.load(subj_dir / "mask.npy")

        # Extract metadata
        row = meta_df[meta_df["sample"] == subj_id]
        if row.empty:
            print(f"Metadata missing for {subj_id}, skipping.")
            continue

        metadata = {
            "v_enc": float(row["v_enc"].values[0]) * -10,
            "pixel_size": float(row["pixel_size"].values[0]),
            "trigger_delay": float(row["delay_trigger"].values[0][1:-1])
        }

        # Compute stroke volumes
        res_pred = compute_flow_and_stroke_volume(phase, mask_pred, metadata)
        res_gt = compute_flow_and_stroke_volume(phase, mask_gt, metadata)

        sv_pred = res_pred["stroke_vol"]
        sv_gt = res_gt["stroke_vol"]
        ratio = min(sv_pred / sv_gt, sv_gt / sv_pred)

        # Save results
        results.append({
            "sample": subj_id,
            "stroke_vol_gt": sv_gt,
            "stroke_vol_pred": sv_pred,
            "similarity_ratio": ratio
        })

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(res_gt["t"], res_gt["flow_corr"], label="GT", marker="o")
        ax[0].plot(res_pred["t"], res_pred["flow_corr"], label="Pred", marker="x")
        ax[0].set_title(f"Flow Curve – {subj_id}")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Flow (mm³/s)")
        ax[0].legend(); ax[0].grid(True)

        delta_t = metadata["trigger_delay"] / 1e3 / len(res_pred["flow_interp"])
        accumulated = np.cumsum(res_pred["flow_interp"]) * delta_t
        ax[1].plot(res_pred["t_interp"], accumulated)
        ax[1].set_title("Accumulated Flow (Pred)")
        ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("Volume (mm³)")
        ax[1].grid(True)

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{subj_id}_flow.png", dpi=150)
        plt.close(fig)

        print(f"Processed {subj_id}: ratio={ratio:.2f}")

    except Exception as e:
        print(f"Error with {subj_id}: {e}")

mean_ratio = np.mean([res["similarity_ratio"] for res in results])
print(f"Mean similarity ratio across all subjects: {mean_ratio:.2f}")
# Save CSV summary
pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"Validation summary written to: {OUT_CSV}")
