"""
Face Recognition-Based Automated Attendance Management System
Streamlit Implementation — Based on: Gomathi S et al., Kongunadu Arts and Science College
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Recognition Attendance", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.metric-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
               padding:1.2rem; text-align:center; }
.metric-value { font-size:2rem; font-weight:700; color:#4f46e5; }
.metric-label { font-size:0.8rem; color:#6b7280; text-transform:uppercase; }
.section-header { font-size:1.1rem; font-weight:600; color:#1e293b;
                  border-left:4px solid #4f46e5; padding-left:0.75rem; margin:1.5rem 0 1rem; }
.info-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:0.8rem 1rem; color:#1e40af; font-size:0.88rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Attendance System")
    st.markdown("*Face Recognition · Deep Learning*")
    st.divider()
    page = st.radio("Navigation", [
        "📊 Dashboard & Overview",
        "📁 Data Upload & EDA",
        "🤖 Model Training & Evaluation",
        "🔬 Simulation & Scenarios",
        "📤 Export Reports",
    ])
    st.divider()
    st.caption("CNN Accuracy: **96.2%** | FAR: **1.8%** | FRR: **2.0%**")

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n_students=40, imgs_per_student=20, noise_level=0.3, seed=42):
    np.random.seed(seed)
    records = []
    for sid in range(1, n_students + 1):
        base_vec = np.random.randn(128)
        for _ in range(imgs_per_student):
            vec = base_vec + np.random.randn(128) * noise_level
            records.append({
                "student_id": f"S{sid:03d}", "label": sid - 1,
                **{f"feat_{i}": vec[i] for i in range(128)},
                "lighting": np.random.choice(["Normal","Low","Bright"], p=[0.6,0.2,0.2]),
                "occlusion": np.random.choice(["None","Partial","Masked"], p=[0.7,0.2,0.1]),
            })
    return pd.DataFrame(records)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    TP = np.diag(cm).sum(); FP = (cm.sum(0)-np.diag(cm)).sum()
    FN = (cm.sum(1)-np.diag(cm)).sum(); TN = cm.sum()-TP-FP-FN
    FAR = FP/(FP+TN) if (FP+TN)>0 else 0
    FRR = FN/(FN+TP) if (FN+TP)>0 else 0
    return acc, FAR, FRR, cm

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard & Overview":
    st.markdown("## Face Recognition Attendance System")
    st.caption("Design and Evaluation using Deep Learning · Kongunadu Arts & Science College")

    c1,c2,c3,c4 = st.columns(4)
    for col, label, val in zip([c1,c2,c3,c4],
        ["CNN Accuracy","False Acceptance Rate","False Rejection Rate","Avg Recognition Time"],
        ["96.2%","1.8%","2.0%","0.8 sec"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                        f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">System Workflow</div>', unsafe_allow_html=True)
    steps = ["📷 Capture","🔍 Detect Faces","🧠 CNN Features",
             "🔎 Match","✅ Mark Attendance","💾 Store & Report"]
    for col, step in zip(st.columns(6), steps):
        col.info(step)

    st.markdown('<div class="section-header">Method Comparison (Paper Results)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7,3.5))
    methods = ["PCA","LBPH","CNN (Proposed)"]
    accs = [85.4, 89.1, 96.2]
    colors = ["#94a3b8","#64748b","#4f46e5"]
    bars = ax.barh(methods, accs, color=colors, height=0.5)
    for b, v in zip(bars, accs):
        ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2, f"{v}%", va="center", fontweight="bold")
    ax.set_xlim(75,101); ax.set_xlabel("Accuracy (%)"); ax.set_title("Recognition Accuracy by Method")
    ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-header">Paper Confusion Matrix</div>', unsafe_allow_html=True)
    paper_cm = np.array([[385,8],[7,400]])
    fig2, ax2 = plt.subplots(figsize=(4,3))
    sns.heatmap(paper_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Present","Pred Absent"],
                yticklabels=["Actual Present","Actual Absent"], ax=ax2)
    fig2.tight_layout(); st.pyplot(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA UPLOAD & EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Data Upload & EDA":
    st.markdown("## Data Upload & Exploratory Analysis")
    tab1, tab2 = st.tabs(["📂 Upload CSV", "🔬 Synthetic Dataset"])

    with tab1:
        uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
            st.dataframe(df.head(20), use_container_width=True)
            st.session_state["df"] = df

    with tab2:
        c1,c2,c3 = st.columns(3)
        n_s = c1.slider("Students", 10, 100, 40)
        i_p = c2.slider("Images / Student", 5, 50, 20)
        noise = c3.slider("Noise Level", 0.1, 1.0, 0.3, 0.05)
        if st.button("🎲 Generate Dataset", type="primary"):
            df = generate_dataset(n_s, i_p, noise)
            st.session_state["df"] = df
            st.success(f"Generated {len(df)} samples for {n_s} students.")

    if "df" in st.session_state:
        df = st.session_state["df"]
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        st.dataframe(df[["student_id","label","lighting","occlusion"]+feat_cols[:5]].head(20), use_container_width=True)

        fig, axes = plt.subplots(1,3,figsize=(14,3.5))
        counts = df["label"].value_counts().sort_index()
        axes[0].bar(counts.index, counts.values, color="#4f46e5", alpha=0.8)
        axes[0].set_title("Samples per Student")
        df["lighting"].value_counts().plot.pie(ax=axes[1], autopct="%1.1f%%",
            colors=["#4f46e5","#7c3aed","#a78bfa"], startangle=90)
        axes[1].set_title("Lighting"); axes[1].set_ylabel("")
        df["occlusion"].value_counts().plot.pie(ax=axes[2], autopct="%1.1f%%",
            colors=["#4f46e5","#f59e0b","#ef4444"], startangle=90)
        axes[2].set_title("Occlusion"); axes[2].set_ylabel("")
        fig.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="section-header">PCA 2D Feature Space</div>', unsafe_allow_html=True)
        pca2 = PCA(n_components=2, random_state=42)
        coords = pca2.fit_transform(StandardScaler().fit_transform(df[feat_cols]))
        sample = df.sample(min(400,len(df)), random_state=42)
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.scatter(coords[sample.index,0], coords[sample.index,1],
                    c=sample["label"], cmap="tab20", alpha=0.6, s=20)
        ax2.set_title("PCA Projection of Face Embeddings"); ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        fig2.tight_layout(); st.pyplot(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training & Evaluation":
    st.markdown("## Model Training & Evaluation")
    if "df" not in st.session_state:
        st.warning("⚠️ Generate/upload a dataset first."); st.stop()

    df = st.session_state["df"]
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    c1,c2,c3 = st.columns(3)
    test_size = c1.slider("Test Split (%)", 10, 40, 30) / 100
    model_choice = c2.selectbox("Model", ["SVM (RBF) ≈ CNN","KNN","PCA+SVM"])
    n_comp = c3.slider("PCA Components", 10, 100, 50) if "PCA" in model_choice else None

    if st.button("🚀 Train & Evaluate", type="primary"):
        X = StandardScaler().fit_transform(df[feat_cols])
        y = df["label"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        with st.spinner("Training..."):
            t0 = time.time()
            if model_choice == "KNN":
                clf = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            elif "PCA" in model_choice:
                pca_m = PCA(n_components=n_comp, random_state=42)
                X_tr = pca_m.fit_transform(X_tr); X_te = pca_m.transform(X_te)
                clf = SVC(kernel="linear", C=1.0)
            else:
                clf = SVC(kernel="rbf", C=10, gamma="scale")
            clf.fit(X_tr, y_tr); y_pred = clf.predict(X_te)
            elapsed = time.time()-t0

        acc, FAR, FRR, cm = compute_metrics(y_te, y_pred)
        st.session_state["eval"] = dict(acc=acc,FAR=FAR,FRR=FRR,cm=cm,elapsed=elapsed)
        st.success("✅ Done!")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc*100:.2f}%", f"{(acc-0.962)*100:+.2f}% vs paper")
        m2.metric("FAR", f"{FAR*100:.2f}%")
        m3.metric("FRR", f"{FRR*100:.2f}%")
        m4.metric("Time", f"{elapsed:.2f}s")

        n_cls = min(cm.shape[0],15)
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        sns.heatmap(cm[:n_cls,:n_cls], ax=axes[0], cmap="Blues",
                    xticklabels=[f"S{i}" for i in range(n_cls)],
                    yticklabels=[f"S{i}" for i in range(n_cls)])
        axes[0].set_title(f"Confusion Matrix (first {n_cls} classes)")

        TP=np.diag(cm).sum(); FN2=(cm.sum(1)-np.diag(cm)).sum()
        FP2=(cm.sum(0)-np.diag(cm)).sum(); TN2=cm.sum()-TP-FN2-FP2
        sns.heatmap(np.array([[TP,FN2],[FP2,TN2]]), annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=["Match","No Match"], yticklabels=["Actual Match","Actual No Match"])
        axes[1].set_title("Binary Confusion")
        fig.tight_layout(); st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(8,3))
        x = np.arange(3); w = 0.35
        ax2.bar(x-w/2,[96.2,1.8,2.0],w,label="Paper",color="#4f46e5",alpha=0.85)
        ax2.bar(x+w/2,[acc*100,FAR*100,FRR*100],w,label=model_choice,color="#7c3aed",alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(["Accuracy","FAR","FRR"])
        ax2.legend(); ax2.set_title("Metrics vs Paper")
        ax2.spines[["top","right"]].set_visible(False); fig2.tight_layout(); st.pyplot(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Simulation & Scenarios":
    st.markdown("## Simulation & Scenario Analysis")
    tab1, tab2, tab3 = st.tabs(["📽️ Attendance Session","🌡️ Condition Impact","📈 Scalability"])

    with tab1:
        c1,c2,c3 = st.columns(3)
        n_sim = c1.slider("Students in class", 10,100,40)
        base_acc = c2.slider("Base Accuracy (%)", 80.0,99.9,96.2,0.1)
        proc_time = c3.slider("Recognition time (s)", 0.3,3.0,0.8,0.1)
        lighting = st.selectbox("Lighting", ["Normal","Low Light","Bright Glare"])
        deg = {"Normal":0,"Low Light":-5.5,"Bright Glare":-2.0}
        eff_acc = (base_acc + deg[lighting]) / 100

        if st.button("▶ Run Session", type="primary"):
            results = []
            prog = st.progress(0)
            for i in range(n_sim):
                recognized = np.random.rand() < eff_acc
                results.append({
                    "Student": f"S{i+1:03d}",
                    "Status": "✅ Present" if recognized else "❌ Absent",
                    "Time (s)": round(proc_time + np.random.uniform(-0.1,0.2),2),
                    "Confidence (%)": round(min(99.9,np.random.normal(eff_acc*100,3)),1),
                })
                prog.progress((i+1)/n_sim)
            prog.empty()
            df_r = pd.DataFrame(results)
            st.session_state["session_results"] = df_r
            present = (df_r["Status"]=="✅ Present").sum()

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Present",present); m2.metric("Absent",n_sim-present)
            m3.metric("Total Time",f"{df_r['Time (s)'].sum():.1f}s")
            m4.metric("Effective Acc",f"{eff_acc*100:.1f}%")
            st.dataframe(df_r, use_container_width=True)

            fig,axes = plt.subplots(1,2,figsize=(12,4))
            axes[0].pie([present,n_sim-present],labels=["Present","Absent"],
                        autopct="%1.1f%%",colors=["#4f46e5","#ef4444"])
            axes[0].set_title("Attendance Distribution")
            cv = df_r["Confidence (%)"].values
            axes[1].hist(cv,bins=20,color="#4f46e5",alpha=0.8,edgecolor="white")
            axes[1].axvline(cv.mean(),color="#ef4444",linestyle="--",label=f"Mean:{cv.mean():.1f}%")
            axes[1].set_xlabel("Confidence (%)"); axes[1].legend()
            axes[1].set_title("Confidence Distribution")
            fig.tight_layout(); st.pyplot(fig)

    with tab2:
        df_light = pd.DataFrame({
            "Condition":["Normal","Low Light","Bright Glare","Flickering","Night"],
            "Accuracy (%)":[96.2,90.7,93.4,88.3,82.1],
            "FAR (%)":[1.8,3.5,2.6,4.2,6.8],
            "FRR (%)":[2.0,5.8,4.1,7.4,11.2],
        })
        fig,axes = plt.subplots(1,2,figsize=(13,4))
        clrs = ["#4f46e5" if c=="Normal" else "#94a3b8" for c in df_light["Condition"]]
        axes[0].bar(df_light["Condition"],df_light["Accuracy (%)"],color=clrs)
        axes[0].set_ylim(70,100); axes[0].axhline(96.2,color="#ef4444",linestyle="--",label="Baseline")
        axes[0].set_title("Accuracy by Lighting"); axes[0].legend()
        x=np.arange(5)
        axes[1].plot(x,df_light["FAR (%)"],marker="o",color="#f59e0b",label="FAR",linewidth=2)
        axes[1].plot(x,df_light["FRR (%)"],marker="s",color="#ef4444",label="FRR",linewidth=2)
        axes[1].set_xticks(x); axes[1].set_xticklabels(df_light["Condition"])
        axes[1].set_title("FAR & FRR by Lighting"); axes[1].legend()
        fig.tight_layout(); st.pyplot(fig)
        st.dataframe(df_light, use_container_width=True)

        df_occ = pd.DataFrame({
            "Occlusion":["None","Sunglasses","Partial Mask","Full Mask","Head Tilt 30°"],
            "Accuracy (%)":[96.2,82.3,74.1,58.6,88.9],
        })
        fig2,ax2 = plt.subplots(figsize=(8,3.5))
        clrs2 = ["#4f46e5" if o=="None" else "#94a3b8" for o in df_occ["Occlusion"]]
        ax2.barh(df_occ["Occlusion"],df_occ["Accuracy (%)"],color=clrs2)
        ax2.set_xlim(40,102)
        for i,v in enumerate(df_occ["Accuracy (%)"]): ax2.text(v+0.5,i,f"{v}%",va="center",fontweight="bold")
        ax2.set_title("Impact of Occlusion on Accuracy")
        ax2.spines[["top","right"]].set_visible(False); fig2.tight_layout(); st.pyplot(fig2)

    with tab3:
        max_s = st.slider("Max Students", 50,1000,500,50)
        sr = np.arange(10, max_s+1, 10)
        acc_c = np.clip(96.2 - 0.008*(sr-40) - 0.000015*(sr-40)**2, 75, 97)
        time_c = 0.8 + 0.0015*sr

        fig,axes = plt.subplots(1,2,figsize=(13,4))
        axes[0].plot(sr,acc_c,color="#4f46e5",linewidth=2)
        axes[0].axvline(40,color="#ef4444",linestyle="--",label="Paper (40 students)")
        axes[0].fill_between(sr,acc_c-1.5,acc_c+1.5,alpha=0.15,color="#4f46e5")
        axes[0].set_xlabel("Students"); axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Accuracy vs Dataset Size"); axes[0].legend()

        axes[1].plot(sr,time_c,color="#7c3aed",linewidth=2)
        axes[1].axhline(0.8,color="#ef4444",linestyle="--",label="Paper baseline (0.8s)")
        axes[1].set_xlabel("Students"); axes[1].set_ylabel("Avg Time (s)")
        axes[1].set_title("Processing Time vs Dataset Size"); axes[1].legend()
        fig.tight_layout(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📤 Export Reports":
    st.markdown("## Export Reports")

    st.markdown('<div class="section-header">Attendance Session CSV</div>', unsafe_allow_html=True)
    if "session_results" in st.session_state:
        df_sess = st.session_state["session_results"]
        st.dataframe(df_sess, use_container_width=True)
        st.download_button("⬇️ Download Attendance CSV",
            df_sess.to_csv(index=False).encode(), "attendance_report.csv","text/csv",type="primary")
    else:
        st.warning("Run a simulation session first.")

    st.markdown('<div class="section-header">Model Performance CSV</div>', unsafe_allow_html=True)
    if "eval" in st.session_state:
        ev = st.session_state["eval"]
        perf = pd.DataFrame([
            {"Model":"Trained","Accuracy (%)":round(ev["acc"]*100,2),
             "FAR (%)":round(ev["FAR"]*100,2),"FRR (%)":round(ev["FRR"]*100,2)},
            {"Model":"Paper CNN","Accuracy (%)":96.2,"FAR (%)":1.8,"FRR (%)":2.0},
        ])
        st.dataframe(perf, use_container_width=True)
        st.download_button("⬇️ Download Performance CSV",
            perf.to_csv(index=False).encode(),"performance_summary.csv","text/csv")
    else:
        st.warning("Train a model first.")

    st.markdown('<div class="section-header">Method Comparison CSV</div>', unsafe_allow_html=True)
    comp = pd.DataFrame([
        {"Method":"RFID","Technology":"ID Card","Accuracy (%)":"—","Drawback":"Proxy possible"},
        {"Method":"Fingerprint","Technology":"Biometric","Accuracy (%)":"~95","Drawback":"Contact-based"},
        {"Method":"PCA","Technology":"ML","Accuracy (%)":85.4,"Drawback":"Lighting sensitive"},
        {"Method":"LBPH","Technology":"ML","Accuracy (%)":89.1,"Drawback":"Low accuracy"},
        {"Method":"CNN (Proposed)","Technology":"Deep Learning","Accuracy (%)":96.2,"Drawback":"High computation"},
    ])
    st.dataframe(comp, use_container_width=True)
    st.download_button("⬇️ Download Comparison CSV",
        comp.to_csv(index=False).encode(),"method_comparison.csv","text/csv")