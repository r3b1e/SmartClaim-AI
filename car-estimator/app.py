"""Web demo: upload a damage photo, enter vehicle/policy details, get a claim estimate.

    ../.venv/bin/python app.py      # then open http://127.0.0.1:7860
"""

from pathlib import Path

import gradio as gr

from estimator import Policy, Vehicle, estimate, vision

EXAMPLES_DIR = vision.REPO / "damage-prediction/training/yolo_dataset/val"


def inr(x):
    return f"₹{x:,.0f}"


def scenario_table(sc):
    rows = ["| Part | Action | Cost |", "|---|---|---:|"]
    rows += [f"| {i.part.replace('_', ' ')} | {i.action} | {inr(i.bill)} |" for i in sc.items]
    rows += [
        f"| **Repair bill (incl. GST)** | | **{inr(sc.repair_bill)}** |",
        f"| − depreciation | | {inr(sc.depreciation)} |",
        f"| − deductible | | {inr(sc.deductible)} |",
        f"| **Insurer pays** | | **{inr(sc.insurer_pays)}** |",
        f"| **You pay** | | **{inr(sc.customer_pays)}** |",
    ]
    if sc.total_loss:
        rows.append("| ⚠️ **Total loss: settled at IDV** | | |")
    return "\n".join(rows)


def report(est, detections):
    zone, level = est.damage_label.split("_")
    where = {"F": "Front", "R": "Rear"}[zone]
    out = [f"### Damage: {where} — {level}  ·  confidence {est.confidence:.0%}"]

    if detections:
        out += ["", "**Damage found**", "", "| Damage | On part | Confidence |", "|---|---|---:|"]
        out += [f"| {d.damage_type.replace('_', ' ')} | {(d.part or '(by zone)').replace('_', ' ')} | {d.confidence:.0%} |"
                for d in detections]

    if est.low and est.high:
        out += ["", f"## Estimated repair: {inr(est.low.repair_bill)} – {inr(est.high.repair_bill)}",
                f"**Insurer pays {inr(est.low.insurer_pays)} – {inr(est.high.insurer_pays)}**, "
                f"you pay {inr(est.low.customer_pays)} – {inr(est.high.customer_pays)}",
                "", "#### Low estimate", scenario_table(est.low),
                "", "#### High estimate", scenario_table(est.high)]
    if not detections and level != "Normal":
        out += ["", "> The damage detector found no specific damage regions in this photo, so the estimate is "
                "based on the location and severity classification."]
    for note in est.notes:
        out.append(f"\n> {note}")
    return "\n".join(out)


def run(image, segment, city_tier, vehicle_age, engine_cc, idv, zero_dep, voluntary_deductible):
    if image is None:
        return None, "Upload a photo of the damaged vehicle."
    label, conf = vision.classify(image)
    detections, results = [], (None, None)
    if vision.default_damage_weights().exists():
        detections, results = vision.detect_damage(image)
    est = estimate(
        label, conf,
        Vehicle(segment, city_tier),
        Policy(vehicle_age, int(engine_cc), idv or None, zero_dep, int(voluntary_deductible)),
        detections,
    )
    drawn = vision.annotate(results)
    return drawn if drawn is not None else image, report(est, detections)


def examples():
    picks = []
    for cls in ("F_Crushed", "F_Breakage", "R_Crushed", "R_Breakage", "F_Normal"):
        folder = EXAMPLES_DIR / cls
        if folder.exists():
            picks.append([str(sorted(folder.iterdir())[3]), "sedan", "tier1", 3, 1200, 600000, False, 0])
    return picks


with gr.Blocks(title="SmartClaim AI") as demo:
    gr.Markdown("# 🚗 SmartClaim AI — Intelligent Vehicle Damage Detection & Insurance Claim Estimator\n"
                "Upload a photo of the damage, enter a few details, and get an instant repair and claim estimate.")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="filepath", label="Damage photo")
            with gr.Accordion("Vehicle & policy details", open=True):
                segment = gr.Dropdown(["hatchback", "sedan", "suv", "luxury"], value="sedan", label="Car segment")
                city_tier = gr.Dropdown(["tier1", "tier2", "tier3"], value="tier1", label="City tier (labour rates)")
                vehicle_age = gr.Slider(0, 15, value=3, step=0.5, label="Vehicle age (years)")
                engine_cc = gr.Number(1200, label="Engine capacity (cc)", precision=0)
                idv = gr.Number(600000, label="Insured Declared Value, IDV (₹)", precision=0)
                zero_dep = gr.Checkbox(False, label="Zero-depreciation add-on")
                voluntary = gr.Number(0, label="Voluntary deductible (₹)", precision=0)
            button = gr.Button("Estimate claim", variant="primary")
        with gr.Column(scale=1):
            annotated = gr.Image(label="Detected damage", interactive=False)
            summary = gr.Markdown()
    inputs = [image, segment, city_tier, vehicle_age, engine_cc, idv, zero_dep, voluntary]
    button.click(run, inputs, [annotated, summary])
    if examples():
        gr.Examples(examples(), inputs, label="Sample photos")
    gr.Markdown("<small>Prices are indicative and depreciation follows the IRDAI schedule; "
                "a surveyor's inspection decides the final claim.</small>")


if __name__ == "__main__":
    demo.launch()
