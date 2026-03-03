import pandas as pd
import numpy as np

print("Generating face recognition dataset...")

# ── Settings (matches paper: 40 students × 20 images = 800 total) ─────────────
N_STUDENTS = 40
IMGS_PER_STUDENT = 20
NOISE_LEVEL = 0.3
SEED = 42

np.random.seed(SEED)

records = []

for sid in range(1, N_STUDENTS + 1):
    # Each student has a unique "base" face embedding (128-dimensional)
    base_vec = np.random.randn(128)
    
    for img_num in range(1, IMGS_PER_STUDENT + 1):
        # Add noise to simulate different photos of same person
        vec = base_vec + np.random.randn(128) * NOISE_LEVEL
        
        row = {
            "student_id": f"S{sid:03d}",
            "student_name": f"Student_{sid:03d}",
            "label": sid - 1,
            "image_number": img_num,
            "lighting": np.random.choice(
                ["Normal", "Low", "Bright"], p=[0.6, 0.2, 0.2]
            ),
            "occlusion": np.random.choice(
                ["None", "Partial", "Masked"], p=[0.7, 0.2, 0.1]
            ),
            "attendance": np.random.choice(
                ["Present", "Absent"], p=[0.85, 0.15]
            ),
        }
        
        # Add 128 facial feature columns (face embedding)
        for i in range(128):
            row[f"feat_{i}"] = round(vec[i], 6)
        
        records.append(row)

# ── Create DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame(records)

# ── Save to CSV ────────────────────────────────────────────────────────────────
output_file = "face_recognition_dataset.csv"
df.to_csv(output_file, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n✅ Dataset generated successfully!")
print(f"📁 Saved as: {output_file}")
print(f"📊 Total rows: {len(df)}")
print(f"👥 Students: {df['student_id'].nunique()}")
print(f"🖼️  Images per student: {IMGS_PER_STUDENT}")
print(f"📐 Total columns: {len(df.columns)}")
print(f"\nColumn overview:")
print(f"  - student_id     : Student identifier (S001 to S040)")
print(f"  - student_name   : Student name")
print(f"  - label          : Numeric class label (0 to 39)")
print(f"  - image_number   : Photo number per student")
print(f"  - lighting       : Lighting condition")
print(f"  - occlusion      : Face occlusion type")
print(f"  - attendance     : Present / Absent")
print(f"  - feat_0..feat_127: 128-dim face embedding features")
print(f"\nFirst 3 rows preview:")
print(df[['student_id','label','lighting','occlusion','attendance','feat_0','feat_1','feat_2']].head(3).to_string())