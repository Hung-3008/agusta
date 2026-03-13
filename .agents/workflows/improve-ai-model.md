---
description: Workflow to systematically research and improve an AI model.
---
// turbo-all

1.  **Define the Problem Scope:** clearly identify the input data, output data (targets/labels), and the specific problem the AI model needs to solve. Detail the current performance metrics and the desired goal.

2.  **Research and Knowledge Gathering:** search for SOTA (State of the Art) methods, academic papers, or existing open-source solutions related to the problem. Compile findings, architecture designs, and potential improvements into a knowledge base document or notes.

3.  **Implementation and Refinement:** write the code to implement the chosen SOTA methods or proposed architecture improvements. Update the model definition, data processing pipelines, or loss functions as necessary.

4.  **Debugging and Testing:** run initial tests to ensure the model compiles and can overfit a single batch (you can use the `debug-ai-model` skill here). Perform full training runs or evaluations to verify the implementation.

5.  **Evaluation and Iteration:** evaluate the model's performance against the goals set in step 1. Analyze the results. If the goals are not met, return to Step 2 to research alternative approaches or refine the current method based on the failure analysis.
