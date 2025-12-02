
# PrivAwareShare: AI-Driven Automated Access Control for Privacy-Preserving Photo Sharing

This repository provides an expanded reference implementation of the **PrivAwareShare**
framework, including:

- PrivNet-CNN: ResNet-based face embedding extractor
- MetaExtractAI: Lightweight demo models for activity, emotion, gesture, and location
- PolicyMatchNet: Privacy compliance scoring module
- ShareDecisionNet: Final sharing decision logic
- PrivAnonGAN: Simple GAN stubs and a practical anonymization demo

The implementation is designed for **reproducibility and demonstration**. It offers a
complete, runnable pipeline that mirrors the logic described in the manuscript. You can
replace the demo components with your own trained models and datasets.

## Quickstart

```bash
pip install -r requirements.txt

python -m privawareshare.utils.io_utils \
    --image_path examples/sample_input.jpg \
    --policies_path examples/sample_policies.json
```

The script prints the final decision (PUBLISH / REQUEST_APPROVAL / ANONYMIZE), along with
intermediate metadata and scores.
