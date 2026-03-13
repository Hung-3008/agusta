---
name: debug-ai-model
description: Skill for debugging AI models, neural networks, or machine learning training pipelines when experiencing poor performance, NaNs, or failing to converge. Includes a systematic checklist for data, forward pass, loss, gradients, and metrics.
---

# Debugging AI Models

When tasked with debugging an AI model that is failing to converge, producing NaNs, or showing poor metrics, follow this systematic debugging process.

## 1. Data Verification
Before investigating the model architecture, ensure the input data is correct.
- Check data shapes and data types (e.g., `float32`, `int64`). You can use snippets to load a batch and print its shape and stats.
- Check data distributions. Look for an abundance of zeros or NaNs in the input.
- Check target/label shapes and ranges. Ensure they align with the loss function's expectations.
- Normalize or standardize inputs if required by the model.

## 2. Overfit a Single Batch
The most crucial test for any model is to see if it can memorize a single batch of data.
- Modify the training script or write a minimal script to run the training loop on the exact same batch repeatedly for 10-50 iterations.
- Turn off data shuffling.
- The training loss should approach zero (or the theoretical minimum).
- If it cannot overfit a single batch, there is a fundamental bug in the model architecture, forward pass, loss calculation, or optimizer.

## 3. Forward Pass and Loss
- Add print statements or use debugging tools to check intermediate tensor shapes and values within the `forward` function.
- Verify the loss function is appropriate for the task (e.g., CrossEntropy for classification, MSE for regression).
- Ensure the loss output is a valid scalar and not NaN/Inf.
- Check for unintended tensor broadcasting in the loss calculation (e.g., `(B,)` vs `(B, 1)`).

## 4. Gradients and Optimization
- After `loss.backward()`, check if gradients are populated (`tensor.grad is not None` and not all zeros/NaNs).
- Verify the learning rate is not too high (causing divergence/NaNs) or too low (causing stalled learning).
- Implement and monitor gradient clipping if loss spiking occurs (e.g., `torch.nn.utils.clip_grad_norm_`).

## 5. Metrics Calculation
- If loss is decreasing but validation metrics (like Pearson correlation) are poor, the metric implementation might be flawed.
- Ensure model outputs are processed correctly before metric calculation (e.g., applying softmax, squeezing dimensions, ensuring both predictions and targets are converted to numpy arrays).

## 6. Common Gotchas
- Forgetting `model.train()` or `model.eval()`.
- Forgetting to call `optimizer.zero_grad()`.
- Using `BCEWithLogitsLoss` but passing probabilities instead of logits.
- Incorrect batch size affecting batch normalization stability.
