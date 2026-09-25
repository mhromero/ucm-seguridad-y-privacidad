from __future__ import annotations

import io

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bundle_io import load_bundle
from utils import format_label_set
from visualizations import _load_prediction_context

matplotlib.use("Agg")


EVAL_BUNDLE_PATH = "artifacts/eval_bundle.pt"
STUDENT_CHECKPOINT_PATH = "artifacts/best_student.pt"

bundle, student_probs, report, label_names = _load_prediction_context(
    eval_bundle_path=EVAL_BUNDLE_PATH,
    student_checkpoint_path=STUDENT_CHECKPOINT_PATH,
)


def _plot_comparison(index: int) -> np.ndarray:
    teacher_probs = bundle["teacher_probs"][index].numpy()
    student_sample = student_probs[index].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    teacher_colors = ["#2ecc71" if value >= 0.5 else "#bdc3c7" for value in teacher_probs]
    student_colors = ["#e74c3c" if value >= 0.5 else "#bdc3c7" for value in student_sample]

    axes[0].barh(label_names, teacher_probs, color=teacher_colors)
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Teacher outputs")

    axes[1].barh(label_names, student_sample, color=student_colors)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Student outputs")
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    buffer.seek(0)
    plot = np.array(Image.open(buffer))
    buffer.close()
    plt.close(fig)
    return plot


def get_random_sample() -> tuple[Image.Image, np.ndarray, str, str, str]:
    index = int(np.random.randint(len(bundle["images"])))
    image = bundle["images"][index, 0].numpy()
    image_pil = Image.fromarray((image * 255).astype(np.uint8), mode="L")

    teacher_binary = (bundle["teacher_probs"][index].numpy() >= 0.5).astype(np.float32)
    student_binary = (student_probs[index].numpy() >= 0.5).astype(np.float32)
    agreement = "MATCH" if np.array_equal(teacher_binary, student_binary) else "DIFF"
    truth = format_label_set(bundle["labels"][index], label_names) if "labels" in bundle else "unknown"
    plot = _plot_comparison(index)

    return (
        image_pil,
        plot,
        agreement,
        truth,
        f"Teacher: {format_label_set(teacher_binary, label_names)} | "
        f"Student: {format_label_set(student_binary, label_names)}",
    )


with gr.Blocks(title="Multilabel Model Extraction Demo") as demo:
    gr.Markdown(
        """
        # Multilabel Model Extraction Demo

        Demo de extracción por distillation sobre imágenes sintéticas con varias figuras por muestra.
        El student solo vio entradas y salidas del teacher, no su arquitectura ni sus pesos.
        """
    )

    with gr.Row():
        input_image = gr.Image(label="Muestra", type="pil", interactive=False, height=220)
        comparison_plot = gr.Image(label="Comparación de probabilidades", height=280)

    with gr.Row():
        agreement_box = gr.Textbox(label="Agreement", interactive=False)
        truth_box = gr.Textbox(label="Etiquetas reales", interactive=False)

    prediction_box = gr.Textbox(label="Resumen", interactive=False)
    sample_button = gr.Button("Random Sample", variant="primary")

    sample_button.click(
        fn=get_random_sample,
        outputs=[input_image, comparison_plot, agreement_box, truth_box, prediction_box],
    )

    gr.Markdown(
        f"""
        Exact agreement global: **{report['teacher_vs_student']['exact_agreement']:.3f}**  
        Labelwise agreement global: **{report['teacher_vs_student']['labelwise_agreement']:.3f}**
        """
    )


if __name__ == "__main__":
    demo.launch()
