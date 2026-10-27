"""Generate the five-page report from recorded experiments. Requires reportlab."""
import json
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'
def load(name):return json.loads((RUN/name).read_text())
r=load('experiment_summary.json');cv=load('cv_results.json');test=load('test-score.json');base=load('test-majority-score.json');timing=load('test-timing.json')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Text',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallText',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=7))
styles['Title'].fontSize=23;styles['Title'].leading=28
styles['Heading1'].fontSize=16;styles['Heading1'].leading=20
styles['Heading2'].fontSize=12;styles['Heading2'].leading=16
story=[]
def p(text,style='Text'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'Heading1')
def sub(text):p(text,'Heading2')
def table(rows,widths):
    header=ParagraphStyle(name='HeaderCell',parent=styles['SmallText'],textColor=colors.white)
    data=[[Paragraph(str(c),header if i==0 else styles['SmallText']) for c in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173D55')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),0.5,colors.HexColor('#173D55')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EEF3F6'),colors.white])]))
    # Paragraph text colors are explicit because Table text styles do not override Paragraph styles.
    for cell in data[0]:cell.style=ParagraphStyle(name='HeaderCell',parent=styles['SmallText'],textColor=colors.white)
    story.append(t);story.append(Spacer(1,12))
def page():story.append(PageBreak())
p('Selecting Reliable LLM Reasoning','Title')
p('DASE7506 Mini-project 2 | Experimental report draft | 27 October 2026','SmallText')
p('Training-only model selection under an informative-test distribution','Heading2')
p('<b>Research question.</b> Can a small candidate-comparison model improve on final-answer majority voting when the available answers disagree? We use fixed supplied representations and lightweight input features, with no external data, extra pretrained weights, or new language-model generation.')
h('1. Evaluation design comes first')
p('The bank contains 4,755 training, 798 validation and 467 test questions, each with four candidate solutions. More than one candidate may be correct. Correctness is defined by the official final numeric answer parser and scorer, not by the validity of every reasoning step. [1, 2]')
p('The published test selection requires at least two distinct valid parsed answers per question. Applying that input-only condition gives 2,114 training and 355 validation hard questions. The remaining 2,641 / 443 questions have only one distinct valid answer. We call this a hard subset for convenience; it matches the stated filter, not every possible aspect of test difficulty.')
table([['Method','Validation all','Validation hard','Test all (hard)'],['Majority vote','688/798 = 86.22%','245/355 = 69.01%','323/467 = 69.16%'],['Frozen 3-seed MLP','740/798 = 92.73%','297/355 = 83.66%','372/467 = 79.66%']],[133,111,116,139])
p('Whole-validation accuracy overstates performance on informative questions: majority voting falls from 86.22% overall to 69.01% on the hard subset. The final method improves test accuracy by 10.49 percentage points (49 net additional correct choices). Test performance remains 4.00 points below hard-validation accuracy; matching the filter does not eliminate all generalization uncertainty.')
sub('Selection and test discipline')
p('Six recipes were compared with three-fold out-of-fold (OOF) predictions inside the training split. Folds keep all four candidates of a question together and are stratified by hard status and number of correct candidates. The winner was selected by pooled hard OOF accuracy. A fixed ensemble of seeds 17, 29 and 43 was then fitted on training data. Official validation was inspected once after recipe selection. Core source files and assets were hashed before the first test score; test results did not change the method.')
p('Status: local implementation, frozen checkpoint and measured scores are available. Student identity, public artifact links and submission Issue remain to be completed. This draft requires student review and understanding before submission.','SmallText')
page()
h('2. Method and training objective')
sub('Input-only features')
table([['Feature block','Dimensions','Purpose'],['Supplied trace representation','1024','Use the fixed numerical representation distributed with the bank.'],['Centered representation','1024','Subtract the mean of the four representations to express relative evidence.'],['Answer validity and consensus','7','Parsed validity, own votes, number of answers, maximum votes, vote gap, invalid count, majority membership.'],['Length and format','6','Log length, relative length, log line/equality counts, #### and boxed indicators.']],[150,65,284])
p('Total input dimension is 2,061 per candidate. Final answers are normalized by the unchanged official parser, so equivalent fractions and decimals share a vote. Invalid answers are excluded from vote counts. All four candidates use the same scoring function; there are no candidate-index, question-ID or lookup-table features.')
p('Each feature is standardized using means and standard deviations fitted only on training questions with both correct and incorrect candidates. Standard deviations are floored at 0.05, and standardized values are clipped to [-10, 10]. In CV, these statistics are re-fitted on each training fold. No validation or test statistics are fitted.')
sub('Shared scoring network')
p('A 2,061-to-64 linear layer, ReLU, dropout 0.2 and a 64-to-1 linear layer produce a scalar score for each candidate. Inference disables dropout and uses exported float32 NumPy arrays. Three independently fitted networks have their raw scores averaged; there is no calibration fitting or learned fusion weight. The selected index is the argmax of those mean scores.')
sub('Multiple correct answers and pairwise loss')
p('Let P be the correct candidates and N the incorrect candidates of a training question, as determined by the official parser. For scores s, the question loss is:<br/><font face="Courier">L = mean over i in P, j in N: softplus(s[j] - s[i])</font>')
p('Loss is averaged across pairs within each question, then across questions. Thus a question with more pairs does not automatically receive greater weight. All-correct questions are excluded because they supply no positive-versus-negative comparison. Every supplied question has a correct candidate. The loss directly learns to rank a correct answer above an incorrect one without requiring a unique correct class.')
p('Training uses AdamW, learning rate 0.001 for MLPs (0.003 for linear models), weight decay 0.03, batch size at most 256, and a fixed 25 epochs. CPU execution uses four intra-operation threads and one inter-operation thread. CV uses seed 17; final members use 17/29/43; fold assignment uses 20261027. There is no early stopping or epoch search in this implementation.')
page()
h('3. Controlled comparisons on training OOF')
p('The experiment budget was six recipes, each evaluated on the same three folds. Epoch count was fixed in advance. Reported OOF values below concern only the 2,114 hard training questions. The three-seed ensemble was a fixed final construction, not one of the CV recipes.')
rows=[['Recipe','Hard correct / 2114','Hard accuracy','Fold range']]
for entry in cv:
    c=entry['config'];m=entry['oof'];folds=entry['fold_hard_accuracy']
    rows.append([c['name'],str(m['hard_correct']),f"{m['hard_accuracy']:.2f}%",f'{min(folds):.2f} - {max(folds):.2f}%'])
table(rows,[163,119,100,117])
p('Majority voting on the same hard training questions achieves 1,413/2,114 = 66.84%. Representation-only linear scoring achieves 78.76%; the context linear model reaches 80.04%. Candidate comparison and consensus therefore provide a useful improvement in this protocol, although the context comparison adds both centered representations and scalar features and cannot isolate their separate effects.')
p('The numerically best recipe is context_mlp at 80.13%. Its advantage over context_linear is only two hard questions (0.095 percentage points), too small to establish that a nonlinear network is superior. The implementation selected the numerical OOF winner; this is a limitation of the selection rule. The linear alternative remains attractive on grounds of simplicity. We did not change the rule after seeing validation or test results.')
sub('Arithmetic and style ablations')
p('The arithmetic extension checks only complete plain arithmetic equations on their own lines. It uses a restricted Python AST visitor supporting numbers, unary signs and +, -, *, /. Unknown expressions, prose and unsupported LaTeX are not marked wrong. Zero division and invalid syntax are ignored. No candidate code is executed.')
p('This deliberately narrow check covers 494/3,192 validation candidates (15.48%). Adding it yields 80.09% hard OOF accuracy, with no improvement over the context MLP. Removing style features while retaining arithmetic gives 79.56%; doubling the weight of hard training questions gives 80.09%. None of these variants justifies replacing the selected recipe. Arithmetic features are absent from the frozen inference model.')
p('The no-style result suggests surface features may contribute, but it does not establish a robust causal explanation: feature dimension and initialization also change, and differences are small. A stronger future study would repeat every ablation over multiple seeds and use paired question-level uncertainty intervals. The public test result must not be used to choose those future variants.')
page()
h('4. Reliability and failure analysis')
sub('Does comparison fix majority errors?')
p('On the 355 hard validation questions, the frozen ensemble corrects 67 majority-vote mistakes and breaks 15 previously correct choices, for a net gain of 52 questions. It is correct on 60.91% of the 110 questions where majority voting is wrong. This shows that the model does more than mechanically repeat consensus, while also showing that overrides carry a real cost.')
rows=[['Valid answer vote pattern','Questions','Correct','Accuracy']]
for pattern,m in r['diagnostics']['groups'].items():rows.append([pattern,m['n'],m['correct'],f"{m['accuracy']:.2f}%"])
table(rows,[190,100,100,109])
p('The pattern omits invalid candidates. Four different valid answers are particularly difficult: 13/20 correct. Counts in several groups are small, so these rates are descriptive and should not be treated as stable subgroup guarantees.')
sub('Permutation and uncertainty checks')
p('One random candidate permutation per validation question preserves a maximum-scoring selection on 100% of questions, allowing numerical tolerance for score ties. Literal selected-candidate agreement is 797/798; the remaining case is tied. This supports the intended position-independent scoring, but is not a proof against all possible shortcuts.')
p('Hard questions split into thirds by the top-two candidate score margin have accuracies 83.19%, 80.51% and 87.29% from low to high margin. The relationship is not monotonic. These margins are not calibrated correctness probabilities. Moreover, two correct candidates with the same answer can have similar scores, making candidate-level margin an imperfect measure of answer-level uncertainty. No confidence-based gate is used.')
sub('Two concrete validation failures')
p('<b>Quiz average.</b> For a third score of 92 and a three-quiz average of 94, the correct first-two average is (3*94 - 92)/2 = 95. The model selects 98 from a fluent trace that incorrectly introduces another score of 92. Its equations look organized but encode the wrong quantities.')
p('<b>Remaining pies.</b> Starting with four pies, selling one and giving away one leaves two pies, not three. A selected trace starts from three, then performs consistent calculations and returns 6 instead of the benchmark answer 4. Local equation checking alone misses the incorrect state transition.')
p('Failure examples and supplied reference reasoning are saved in runs/validation_failure_examples.json. Some bank wording can also be ambiguous; benchmark correctness must be distinguished from a claim that every alternate interpretation is mathematically indefensible. No test errors were used for development.')
page()
h('5. Reproduction, resources and limits')
table([['Measurement','Recorded value'],['Inference assets, uncompressed',f"{r['asset_bytes']:,} bytes ({r['asset_bytes']/2**20:.3f} MiB), including config and all 3 members"],['Training and CV, local CPU',f"{r['training_seconds']:.2f} seconds for the six-recipe search and final fitting/evaluation"],['Test prediction + independent scoring',f"{timing['total_wall_seconds']:.3f} seconds; two fresh processes, four threads"],['Local environment','Windows 11; Intel64 Family 6 Model 154 Stepping 3; Python 3.12; NumPy 2.5.3; PyTorch 2.7.1+cu126 (CPU execution)']],[181,318])
p('Inference imports NumPy, not PyTorch. Resource timing includes process startup, imports, official input checks/loading, feature construction, all ensemble members and independent scoring. The reference Xeon Platinum 8457C has not been measured; local timing does not independently certify reference-machine eligibility. Assets are below the 16 MiB cap with substantial margin.')
sub('Use the frozen checkpoint without training')
p('From code/, install requirements-inference.txt, then run:<br/><font face="Courier">python check_data.py<br/>python verify_features.py<br/>python benchmark.py --split validation<br/>python benchmark.py --split test</font>','SmallText')
p('The benchmark invokes the unchanged predict.py and score_predictions.py with method.py and runs/model_bundle. Official scores are runs/validation-score.json and runs/test-score.json. Hashes in runs/freeze_manifest.json identify the inference sources and assets that were frozen before test scoring. A fresh audit bank is accepted through the official runner and its --data-dir argument; the method reads bank/arrays arguments and does not open reference files.')
sub('Train from scratch')
p('Install requirements-training.txt and run:<br/><font face="Courier">python train.py --epochs 25 --output runs_reproduced</font><br/>This preserves the frozen checkpoint. Use the official runner with --checkpoint runs_reproduced/model_bundle for validation. Dependency and platform differences can affect numerical training results; checkpoint evaluation is the primary score-reproduction path.','SmallText')
sub('Limitations and conclusion')
p('The held-out bank supports a gain over the fixed majority baseline, not a guarantee of mathematical verification or universal out-of-distribution generalization. Only three CV folds and one CV training seed were used. Nonlinear-model superiority is not established, margins are not calibrated, arithmetic coverage is narrow, and no separate audit bank has been evaluated. The strongest practical finding is that a compact learned comparator can improve informative-question selection while easy-question averages conceal substantial remaining error.')
sub('AI assistance and sources')
p('OpenAI Codex provided substantive assistance with assignment/data inspection, experimental design, code, training/evaluation and this report draft. The student must review and understand the implementation and claims. Official data, loaders, parser and scorer are reused unchanged with retained attribution. No external datasets or additional pretrained weights were used.','SmallText')
p('[1] Supplied guide/GUIDE.md, DASE7506 MP2, protocol dase7506-mp2-open-hard-verifier-v5.<br/>[2] Supplied code/README.md and common.py (official final-answer parser/scorer).<br/>[3] GSM8K attribution and retained MIT license: code/data/LICENSE.txt and NOTICE.txt; candidate model provenance: code/data/manifest.json.<br/>Repository: https://github.com/xudongwu-0/DASE7506-MP2','SmallText')
def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#CBD5DC'));canvas.line(48,39,A4[0]-48,39)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526775'));canvas.drawString(48,27,'DASE7506 MP2 | Reviewed student authorship pending | Report draft');canvas.drawRightString(A4[0]-48,27,str(doc.page))
output=ROOT.parent/'report-draft.pdf'
doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=52)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(output)
