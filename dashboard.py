
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Communication-Efficient Federated Learning",
    page_icon="🌐",
    layout="wide"
)

# ============================================================
# FROZEN EXPERIMENT RESULTS
# ============================================================

STANDARD = {
    "accuracy": 96.96,
    "precision": 85.42,
    "recall": 98.91,
    "f1": 91.67,
    "updates_sent": 100,
    "updates_skipped": 0,
    "communication": 2202000,
    "latency": 1.7616
}

SELECTIVE = {
    "accuracy": 95.09,
    "precision": 77.97,
    "recall": 98.97,
    "f1": 87.22,
    "updates_sent": 40,
    "updates_skipped": 60,
    "communication": 880800,
    "latency": 0.7046
}

# ============================================================
# ROUND-BY-ROUND RESULTS
# ============================================================

rounds = list(range(1, 21))

standard_accuracy = [
    94.09, 96.07, 94.80, 96.70, 96.84,
    96.58, 95.76, 95.46, 95.68, 97.21,
    95.95, 97.02, 95.66, 96.68, 96.83,
    96.87, 97.59, 97.20, 96.81, 96.96
]

standard_f1 = [
    84.69, 89.06, 86.51, 90.73, 91.13,
    90.73, 88.74, 88.09, 88.56, 92.09,
    89.19, 91.81, 88.52, 91.00, 91.35,
    91.43, 93.26, 92.28, 91.35, 91.67
]

selective_accuracy = [
    94.09, 96.07, 94.80, 96.70, 96.84,
    96.58, 95.76, 97.17, 95.09,
    95.09, 95.09, 95.09, 95.09, 95.09,
    95.09, 95.09, 95.09, 95.09, 95.09,
    95.09
]

selective_f1 = [
    84.69, 89.06, 86.51, 90.73, 91.13,
    90.73, 88.74, 92.18, 87.22,
    87.22, 87.22, 87.22, 87.22, 87.22,
    87.22, 87.22, 87.22, 87.22, 87.22,
    87.22
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2.2rem 2.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef5ff, #f8fbff);
    border: 1px solid #dce8f7;
    margin-bottom: 1.5rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
}

.hero p {
    margin-top: 0.5rem;
    color: #526070;
    font-size: 1.08rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1.6rem;
    margin-bottom: 0.8rem;
}

.metric-card {
    padding: 1.1rem 1.2rem;
    border: 1px solid #e3e8ef;
    border-radius: 14px;
    background: white;
    min-height: 110px;
}

.metric-label {
    color: #667085;
    font-size: 0.88rem;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    margin-top: 0.25rem;
}

.metric-note {
    color: #667085;
    font-size: 0.78rem;
    margin-top: 0.25rem;
}

.workflow-card {
    padding: 1rem;
    border: 1px solid #e1e7ef;
    border-radius: 12px;
    background: #fafcff;
    min-height: 105px;
}

.workflow-number {
    color: #667085;
    font-size: 0.75rem;
    font-weight: 600;
}

.workflow-title {
    font-weight: 700;
    margin-top: 0.25rem;
}

.workflow-description {
    color: #667085;
    font-size: 0.78rem;
    margin-top: 0.3rem;
}

.finding {
    padding: 1.25rem;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.finding-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.5rem;
}

.finding-text {
    color: #475467;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #667085;
    font-size: 0.8rem;
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def workflow_card(number, title, description):
    st.markdown(
        f"""
        <div class="workflow-card">
            <div class="workflow-number">{number}</div>
            <div class="workflow-title">{title}</div>
            <div class="workflow-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>Communication-Efficient Federated Learning</h1>
    <p>
        QoS-Oriented Federated Learning in a 5G-Inspired Environment
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# OBJECTIVE
# ============================================================

st.markdown(
    '<div class="section-title">Project Objective</div>',
    unsafe_allow_html=True
)

st.info(
    "Reduce unnecessary federated model transmissions by sending only "
    "significant client updates, while evaluating the resulting "
    "communication-performance trade-off against standard FedAvg."
)

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

st.markdown(
    '<div class="section-title">Experiment Configuration</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    metric_card("Dataset", "CICIDS2017", "Network intrusion traffic")

with c2:
    metric_card("Clients", "5", "Simulated participants")

with c3:
    metric_card("Rounds", "20", "Federated rounds")

with c4:
    metric_card("Local Epochs", "2", "Per client / round")

with c5:
    metric_card("Threshold", "0.70", "Update-norm threshold")

# ============================================================
# FEDERATED LEARNING WORKFLOW
# ============================================================

st.markdown(
    '<div class="section-title">Federated Learning Workflow</div>',
    unsafe_allow_html=True
)

workflow = [
    ("01", "CICIDS2017", "Network traffic dataset"),
    ("02", "Preprocessing", "Cleaning, binary target and scaling"),
    ("03", "5 Clients", "Distributed training data"),
    ("04", "Local Training", "Two local epochs"),
    ("05", "Update Evaluation", "Calculate update norm"),
    ("06", "Selective Communication", "Transmit or skip"),
    ("07", "FedAvg", "Aggregate updates"),
    ("08", "Global Model", "Redistribute model"),
    ("09", "Evaluation", "Global test set")
]

for i in range(0, len(workflow), 3):

    cols = st.columns(3)

    for col, item in zip(cols, workflow[i:i + 3]):

        with col:
            workflow_card(
                item[0],
                item[1],
                item[2]
            )

    st.write("")

# ============================================================
# MODEL
# ============================================================

st.markdown(
    '<div class="section-title">Model Architecture</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    metric_card(
        "Architecture",
        "52 → 64 → 32 → 1",
        "PyTorch MLP with ReLU"
    )

with m2:
    metric_card(
        "Parameters",
        "5,505",
        "Trainable parameters"
    )

with m3:
    metric_card(
        "Update Size",
        "22,020 B",
        "Float32 payload"
    )

# ============================================================
# COMPARISON TABLE
# ============================================================

st.markdown(
    '<div class="section-title">Standard FedAvg vs Selective FedAvg</div>',
    unsafe_allow_html=True
)

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Updates Sent",
        "Updates Skipped",
        "Communication",
        "Simulated Latency"
    ],
    "Standard FedAvg": [
        "96.96%",
        "85.42%",
        "98.91%",
        "91.67%",
        "100",
        "0",
        "2,202,000 B",
        "1.7616 s"
    ],
    "Selective FedAvg": [
        "95.09%",
        "77.97%",
        "98.97%",
        "87.22%",
        "40",
        "60",
        "880,800 B",
        "0.7046 s"
    ],
    "Change": [
        "−1.87 pp",
        "−7.45 pp",
        "+0.06 pp",
        "−4.45 pp",
        "60% fewer",
        "+60",
        "60% reduction",
        "60% reduction"
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# COMMUNICATION EFFICIENCY
# ============================================================

st.markdown(
    '<div class="section-title">Communication Efficiency</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "Possible Updates",
        "100",
        "5 clients × 20 rounds"
    )

with c2:
    metric_card(
        "Updates Sent",
        "40",
        "Selective FedAvg"
    )

with c3:
    metric_card(
        "Updates Skipped",
        "60",
        "Selective FedAvg"
    )

with c4:
    metric_card(
        "Communication Reduction",
        "60%",
        "1,321,200 bytes saved"
    )

# ============================================================
# COMMUNICATION GRAPH
# ============================================================

st.markdown("#### Communication Volume")

fig, ax = plt.subplots(figsize=(9, 4))

ax.bar(
    ["Standard FedAvg", "Selective FedAvg"],
    [
        STANDARD["communication"],
        SELECTIVE["communication"]
    ]
)

ax.set_ylabel("Bytes Transmitted")
ax.set_title("Model Communication")

ax.grid(
    axis="y",
    alpha=0.25
)

st.pyplot(fig, clear_figure=True)

# ============================================================
# PERFORMANCE GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">Performance Across Federated Rounds</div>',
    unsafe_allow_html=True
)

metric_choice = st.radio(
    "Select metric",
    ["Accuracy", "F1 Score"],
    horizontal=True
)

fig, ax = plt.subplots(figsize=(11, 4.8))

if metric_choice == "Accuracy":

    ax.plot(
        rounds,
        standard_accuracy,
        marker="o",
        label="Standard FedAvg"
    )

    ax.plot(
        rounds,
        selective_accuracy,
        marker="o",
        label="Selective FedAvg"
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Across 20 Federated Rounds")

else:

    ax.plot(
        rounds,
        standard_f1,
        marker="o",
        label="Standard FedAvg"
    )

    ax.plot(
        rounds,
        selective_f1,
        marker="o",
        label="Selective FedAvg"
    )

    ax.set_ylabel("F1 Score (%)")
    ax.set_title("F1 Score Across 20 Federated Rounds")

ax.set_xlabel("Federated Round")
ax.set_xticks(rounds)

ax.grid(alpha=0.25)
ax.legend()

st.pyplot(fig, clear_figure=True)

# ============================================================
# SIMULATED LATENCY
# ============================================================

st.markdown(
    '<div class="section-title">Simulated Communication Latency</div>',
    unsafe_allow_html=True
)

l1, l2 = st.columns(2)

with l1:
    metric_card(
        "Standard FedAvg",
        "1.7616 s",
        "100 model updates"
    )

with l2:
    metric_card(
        "Selective FedAvg",
        "0.7046 s",
        "40 model updates"
    )

st.caption(
    "Latency is estimated as payload bytes ÷ 10 Mbps. "
    "This is a simulated transmission estimate, not measured physical 5G latency."
)

# ============================================================
# KEY FINDING
# ============================================================

st.markdown(
    '<div class="section-title">Key Finding</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="finding">

<div class="finding-title">
Communication-Performance Trade-off
</div>

<div class="finding-text">

At an update-norm threshold of <b>0.70</b>, Selective FedAvg transmitted
<b>40 of 100 possible client updates</b>. This resulted in a
<b>60% reduction in modeled communication</b> and a
<b>60% reduction in simulated payload latency</b>.

The reduction came with lower final accuracy and F1 compared with Standard
FedAvg, while recall remained almost unchanged
(<b>98.91% → 98.97%</b>).

</div>

</div>
""", unsafe_allow_html=True)

# ============================================================
# LIMITATIONS
# ============================================================

st.markdown(
    '<div class="section-title">Scope & Limitations</div>',
    unsafe_allow_html=True
)

st.markdown("""
- This is a **5G-inspired simulation**, not a physical 5G deployment.
- Latency is modeled using a **10 Mbps bandwidth assumption**.
- Protocol and serialization overhead are not included in communication size.
- The experiment uses **binary classification: Normal vs Attack**.
- A common scaler is fitted using the client-training data.
- Multiclass classification and additional FL algorithms are future extensions.
""")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">
Communication-Efficient Federated Learning
&nbsp;•&nbsp;
CICIDS2017
&nbsp;•&nbsp;
PyTorch
&nbsp;•&nbsp;
5G-Inspired Simulation
</div>
""", unsafe_allow_html=True)

