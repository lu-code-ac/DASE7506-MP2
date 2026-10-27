# Mini-project 2: **Selecting Reliable LLM Reasoning**

**Course:** DASE7506 Advanced Machine Learning Methods  
**Author:** 3036804397  
**Date:** 27 October 2026  
**Final method:** An ensemble of shared candidate-ranking MLPs trained with three random seeds  
**Official test accuracy:** 79.65738758029978% (372/467)

## Abstract

This project investigates how to select a candidate with the correct final numeric answer from four solutions to the same mathematics problem. The training and validation sets contain many easy questions, whereas each test question must contain at least two distinct valid parsed answers. Overall validation accuracy can therefore overestimate performance on difficult questions. We use the validation subset with answer disagreement as the primary development metric and train a lightweight candidate ranker using the supplied fixed representations, answer consensus, and a small number of formatting features. The final model achieves 92.73%, 83.66%, and 79.66% accuracy on the full validation set, hard validation subset, and official test set, respectively. The corresponding majority-vote baselines are 86.22%, 69.01%, and 69.16%. Follow-up experiments with correctness supervision, answer aggregation, set networks, random Fourier features, lexical residuals, and a marginal-likelihood objective did not demonstrate an improvement over the original model, which was therefore retained. The experiments show that higher training-based scores do not necessarily imply better generalization to a difficult distribution.

## 1. Task and Evaluation Protocol

Each example contains a question, four candidate solutions, and fixed representations of shape `[4, 1024]`. The model must return one candidate index from 0 to 3 and cannot generate a new solution. Evaluation uses the official `common.parse_answer` function to normalize final numeric answers, including equivalent fractions and decimals. Multiple candidates may be correct. A candidate with flawed reasoning can still receive credit if its final answer is correct; the task therefore evaluates answer selection rather than step-by-step proof verification.

The dataset contains 4,755 training, 798 validation, and 467 test questions. Test selection requires at least two distinct valid parsed answers per question. Using only the candidate final answers in the inputs, we construct a hard subset with the same condition:

| Split      | Total questions |                                        Hard subset | Remaining questions |
| ---------- | --------------: | -------------------------------------------------: | ------------------: |
| Training   |           4,755 |                                              2,114 |               2,641 |
| Validation |             798 |                                                355 |                 443 |
| Test       |             467 | All questions, according to the published protocol |                   0 |

Here, “hard” refers to the published selection condition; it does not imply that every aspect of semantic difficulty has been aligned. Always selecting candidate 0 achieves 78.20% on the full validation set but only 54.93% on the hard subset. Majority voting achieves 86.22% and 69.01%, respectively. We therefore report both full-validation and hard-subset performance, using the latter as the primary selection criterion.

For the original method, three-fold cross-validation was first performed entirely within the training set. All four candidates for a question remained in the same fold, with stratification by hard-subset membership and the number of correct candidates. After selecting a recipe by hard-subset out-of-fold (OOF) accuracy, we retrained it with three random seeds, inspected official validation performance, froze hashes of the core source files and inference assets, and finally evaluated on the test set. Test labels were not used in features, training, lookup tables, or model selection.

Follow-up improvement experiments were conducted after the original test score was known. Development still used only training OOF predictions and the official validation set. No additional candidate was scored on the test set. These follow-up results do not constitute new independent test evidence.

## 2. Features and Model

### 2.1 Input Features

The final model constructs a 2,061-dimensional input for each candidate:

| Feature block                           | Dimensions | Definition or purpose                                        |
| --------------------------------------- | ---------: | ------------------------------------------------------------ |
| Supplied fixed representation           |      1,024 | Use the provided whole-solution representation               |
| Within-question centered representation |      1,024 | Subtract the mean representation of the four candidates from the current representation |
| Answer validity and consensus           |          7 | Parseability, votes for the current answer, number of distinct valid answers, maximum vote count, gap between the two largest vote counts, number of invalid candidates, and majority-answer membership |
| Length and formatting                   |          6 | Log text length, relative length, log newline count, log equals-sign count, and indicators for `####` and `\boxed` |

Votes are based on answers normalized by the official parser, so `1/2` and `0.5` are not treated as different answers. Unparseable answers do not receive consensus votes. The model does not use candidate positions, question IDs, or example lookup tables.

Per-feature means and standard deviations are fitted only on training questions containing both correct and incorrect candidates. Standard deviations are floored at 0.05, and standardized values are clipped to `[-10, 10]`. Statistics are fitted independently within each training fold during cross-validation. Computing the mean of the four candidates within the current question uses the current input only; it does not fit statistics across the validation set.

### 2.2 Shared Scoring Network

All four candidates share the same network:

```text
2061-dimensional input → Linear(64) → ReLU → Dropout(0.2) → Linear(1)
```

The network produces one scalar score per candidate. Three independent members are trained, their raw scores are averaged, and the candidate with the highest mean score is selected. Inference uses exported float32 arrays and NumPy without loading PyTorch.

### 2.3 Ranking Objective with Multiple Positive Candidates

Let P denote the set of correct candidates for a question, N the set of incorrect candidates, and s_i the score of candidate i. The per-question loss is:

$$
L_q=\frac{1}{|P||N|}\sum_{i\in P}\sum_{j\in N}\log(1+\exp(s_j-s_i)).
$$

Losses are first averaged over all positive-negative pairs within each question and then across questions, preventing questions with more pairs from receiving greater weight. This objective supports multiple correct candidates rather than arbitrarily treating one as the only correct class. Questions for which all candidates are correct provide no ranking supervision and are excluded from the original training procedure.

Training uses AdamW with an MLP learning rate of 0.001, weight decay of 0.03, dropout of 0.2, batches of at most 256 questions, and a fixed 25 epochs. The fold-assignment seed is 20261027, the initial search training seed is 17, and the final member seeds are 17, 29, and 43. The original procedure uses neither an epoch-count search nor early stopping.

## 3. Initial Experiments and Ablations

The following results use training OOF predictions on the hard subset, with the same three folds and the same training seed. All denominators are 2,114:

| Method                                                       | Correct questions |   Accuracy |
| ------------------------------------------------------------ | ----------------: | ---------: |
| Majority vote                                                |             1,413 |     66.84% |
| Linear model using supplied representations only             |             1,665 |     78.76% |
| Linear model with representations and candidate-comparison features |             1,692 |     80.04% |
| Candidate-comparison MLP                                     |             1,694 | **80.13%** |
| Add narrow-coverage arithmetic features                      |             1,693 |     80.09% |
| Remove formatting features while retaining arithmetic features |             1,682 |     79.56% |
| Double hard-question weights in the arithmetic-feature model |             1,693 |     80.09% |

The supplied representations substantially outperform simple consensus, and adding candidate comparison and consensus provides a further gain. However, the selected MLP differs from the linear model by only two questions, which is insufficient to establish a stable advantage for the nonlinear architecture. The comparison features add both centered representations and scalar features, so their effects cannot be attributed separately.

The initial arithmetic checker examines only complete, standalone lines containing simple equations. Unsupported expressions are marked unknown rather than incorrect. It covers only 494/3,192 validation candidates and provides no OOF improvement, so arithmetic features are excluded from the final model. The decrease after removing formatting features does not establish that the model has learned mathematical correctness: input dimensionality and initialization also change, and the difference is small.

## 4. Final Results and Reliability

### 4.1 Official Scores

| Method                        | Full validation set | Hard validation subset |   Official test set |
| ----------------------------- | ------------------: | ---------------------: | ------------------: |
| Majority vote                 |     688/798, 86.22% |        245/355, 69.01% |     323/467, 69.16% |
| Final three-seed MLP ensemble | **740/798, 92.73%** |    **297/355, 83.66%** | **372/467, 79.66%** |

The final model answers 49 additional test questions correctly compared with majority voting, a gain of 10.49 percentage points. Hard-validation accuracy remains approximately 4.00 percentage points higher than test accuracy, indicating that matching the selection condition does not eliminate all distribution differences.

The three individual members achieve hard-validation accuracies of 82.54%, 83.38%, and 82.82%, while their mean-score ensemble achieves 83.66%. This observation supports the practical value of ensembling here but does not guarantee a comparable gain on other question banks.

### 4.2 Correcting and Breaking Majority-Vote Decisions

Across the 355 hard validation questions, the final model corrects 67 majority-vote errors but also changes 15 previously correct decisions into errors, producing a net gain of 52 correct answers. It is correct on 60.91% of the 110 questions where majority voting is wrong. The model can therefore improve on simply following consensus, although overriding the majority also has a cost.

| Valid-answer vote pattern        | Questions | Correct selections by final model | Accuracy |
| -------------------------------- | --------: | --------------------------------: | -------: |
| 3–1                              |       151 |                               137 |   90.73% |
| 2–2                              |        34 |                                31 |   91.18% |
| 2–1–1                            |        88 |                                67 |   76.14% |
| 1–1–1–1                          |        20 |                                13 |   65.00% |
| 2–1, with an invalid candidate   |        28 |                                24 |   85.71% |
| 1–1–1, with an invalid candidate |        20 |                                14 |   70.00% |
| 1–1, with invalid candidates     |        14 |                                11 |   78.57% |

Questions with four distinct answers are particularly difficult, but this group contains only 20 questions. These subgroup rates should not be interpreted as stable performance guarantees.

### 4.3 Candidate Permutations and Uncertainty

After randomly permuting the candidates and their corresponding representations for each validation question, selection consistency is 100% when allowing numerical tolerance and equivalent maximum-score choices. Literal agreement after mapping selected indices back is 797/798; the remaining question has tied scores. This supports position-independent behavior but does not rule out all other shortcuts.

Dividing hard questions into three groups by the gap between the highest and second-highest candidate scores gives accuracies of 83.19%, 80.51%, and 87.29%, which are not monotonic. The score gap is therefore not a calibrated probability of correctness. Different correct candidates may also receive similar scores, and answer-aggregation models can assign tied scores to candidates sharing an answer. A candidate-level margin should not be interpreted directly as answer-level confidence.

### 4.4 Failure Cases

**Quiz average:** A third quiz score of 92 and an average of 94 across three quizzes imply a first-two average of `(3*94-92)/2 = 95`. The model instead selects 98 from a solution that incorrectly introduces another score of 92. Well-formed equations do not guarantee correct variable meanings.

**Remaining pies:** Starting with four pies, selling one and giving one away leaves two. The selected solution incorrectly starts from three remaining pies, after which some calculations are internally consistent, producing 6 rather than the reference answer of 4. Local arithmetic checking cannot replace understanding the question and tracking state changes.

Examples are stored in `code/runs/validation_failure_examples.json`. Some question wording may also be ambiguous; disagreement with the reference answer does not imply that every alternative linguistic interpretation is invalid.

## 5. Follow-up Improvement Attempts

The follow-up study compares 37 training configuration designs and records 510 primary combinations when epoch counts and aggregation rules are included. These combinations are highly correlated, and the best search score is subject to selection bias; they are not 510 independent pieces of evidence. No additional model is scored on the test set.

| Follow-up method                                             | Hard-validation accuracy | Correct questions |
| ------------------------------------------------------------ | -----------------------: | ----------------: |
| Retained original model                                      |               **83.66%** |       **297/355** |
| Correctness supervision and answer-aggregation ensemble      |                   81.97% |           291/355 |
| Four-candidate set-comparison network                        |                   82.54% |           293/355 |
| Random Fourier nonlinear features                            |                   81.69% |           290/355 |
| Hashed unigram and bigram lexical residual                   |                   83.38% |           296/355 |
| Answer-probability fusion of the original and lexical models |                   82.54% |           293/355 |

Correctness supervision allows the use of all-correct questions that contribute no ranking loss in the original method. Combined with stronger regularization and answer aggregation, the best initial single-seed OOF accuracy reaches 81.84%, and an initially selected ensemble reaches 82.36%. Three-seed confirmation yields 81.98%, compared with 80.56% for a matched three-seed OOF baseline of the original model. However, hard-validation accuracy is only 81.97%, so the OOF gain does not transfer to validation.

The extended arithmetic checker supports some LaTeX fractions, inline equations, and decimal rounding. It covers 15,282/19,020 training candidates, or 80.35%. It flags equation inconsistencies in 1,071/14,353 candidates with correct final answers and 525/4,667 incorrect candidates. These are checker flags rather than manually verified errors; their limited separation does not justify using them as hard rejection rules.

The set network uses a shared encoder together with candidate means, maxima, and same-answer neighborhood aggregation, but shows no stable advantage. Random Fourier features provide another regularized nonlinear mapping without improving validation performance. The lexical residual learns hashed unigram and bigram weights from scratch, without an additional pretrained model, but answers one fewer validation question correctly than the original model and does not establish an improvement.

Finally, a multiple-positive marginal-likelihood objective directly optimizes the probability mass assigned to at least one correct candidate. Its five configurations achieve at most 79.80% hard OOF accuracy and are not promoted to further validation or test evaluation.

These results do not establish that all complex methods are ineffective. Rather, the training-based gains found in this search lack stable validation support. We therefore retain the original model and make no unverified claim of improved test performance.

## 6. Resources and Reproduction

### 6.1 Measurements

| Item                                     | Local measurement                                            |
| ---------------------------------------- | ------------------------------------------------------------ |
| Recommended model inference assets       | 1,638,571 bytes, approximately 1.563 MiB, including all three members and configuration |
| Original complete experiment runtime     | Approximately 86.36 seconds, including the initial search, final fitting, and associated evaluation |
| Test prediction plus independent scoring | Approximately 0.713 seconds, using two fresh processes and four threads |
| Operating system and processor           | Windows 11; Intel64 Family 6 Model 154 Stepping 3            |
| Python and dependencies                  | Python 3.12; NumPy 2.5.3; PyTorch 2.7.1+cu126                |
| Original training and inference device   | CPU                                                          |
| Follow-up training devices               | CPU and a single NVIDIA GeForce RTX 3050 Ti Laptop GPU       |

Inference timing includes process startup, imports, official data checking and loading, preprocessing, all three members, and independent scoring. The designated Xeon Platinum 8457C has not been measured, so local timing cannot substitute for an eligibility determination on the reference machine. Additional experimental checkpoints are not required inference assets for the recommended model.

### 6.2 Reproduction Without Retraining

After extracting the project, enter `code/` and run the following in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-inference.txt
python check_data.py
python verify_features.py
python benchmark.py --split validation
python benchmark.py --split test
```

The recommended checkpoint is already located in `code/runs/model_bundle/`. The final inference entry point is `code/method.py`; do not mix other experimental models into this checkpoint directory. `benchmark.py` sets thread limits and invokes the unchanged official prediction runner and scorer.

To run the official commands separately:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python predict.py --method method.py --checkpoint runs/model_bundle --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

The score to submit is the `accuracy` field in `code/runs/test-score.json`, namely **79.65738758029978**, not full-validation or training OOF accuracy. Repeated execution of these test commands is for reproducing the frozen result, not for further tuning.

### 6.3 Training from Scratch

```powershell
python -m pip install -r requirements-training.txt
python train.py --epochs 25 --output runs_reproduced
```

The separate output directory avoids overwriting the frozen checkpoint. Training results may vary numerically across platforms or dependencies; the supplied checkpoint should be used preferentially to reproduce the reported score. Frozen version and asset hashes are recorded in `code/runs/freeze_manifest.json` and `code/runs/asset_hashes.json`. Follow-up experiment commands and directory descriptions are provided in `code/README.md`, with a summary in `code/runs_upgrade/followup_summary.json`.

The method uses the supplied `bank` and `arrays` arguments, does not read reference answers, and does not depend on fixed sample counts or question IDs. Additional checks verified behavior with new IDs, question subsets, repeated questions, and reordered inputs.

## 7. Limitations and Conclusion

The method substantially outperforms the fixed majority-vote baseline but is not a mathematical reasoning verifier and does not guarantee equivalent performance on unseen question types. The initial three-fold search uses only one training seed; the larger follow-up search is subject to selection bias; several subgroups are small; candidate margins are uncalibrated; and no independent audit-bank results are available. The initial MLP gain over the linear model is also too small to support a strong conclusion.

The most reliable finding is that a high proportion of easy questions can conceal errors under answer disagreement, while a lightweight learned comparator can improve selection at low computational cost. At the same time, adding supervision types, model complexity, or training-based score improvements does not automatically improve hard-validation performance. The frozen original model is retained, with unsuccessful follow-up experiments preserved as evidence for the generalization analysis.

## 8. AI Assistance and Source Disclosure

This project was completed with assistance from OpenAI Codex. The official data, loaders, numeric parser, and scorer retain their original implementations. No external datasets or additional pretrained weights were used. GSM8K attribution, candidate-model provenance, and licensing information are retained in `code/data/LICENSE.txt`, `NOTICE.txt`, and `manifest.json`. The candidates and fixed representations were supplied by the course and originate from Qwen3-0.6B.