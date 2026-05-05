# Final Report Rubric, Requirements, and Codebase Execution Prompt

## Suggested outline for the final report

**Suggested outline for the final report.**

**Introduction:** Introduction of the problem and the data/environment you are using.

Introduction should include:

- Problem introduction:  Significance of the problem (why it's important to solve it for intellectual and/or social, economic reasons etc.).
- Problems should be stated in terms of what needs to be solved by the agent.
- What is assumed known or given as background to the agent or problem and how the agent.
- What background is there to the problem - contextualize the problem in terms of relevant literature.
- Why is this problem a "problem" - what characteristics make it challenging or fruitful. Many AI problems are smaller learning/inference/optimization problems embedded in an overall agent's design- make sure you describe the overall problem and the key subproblem of interest.
- How is the data or environment appropriate?  How will the agent interact with your simulation or environment?
- Sketch your overall solution strategy.

**Methods and Algorithms:**

Clearly describe the methods and algorithms used in the projects including any feature engineering, any (external) tools or pre-trained models you used. By ‘clearly’ I mean illustrate your understanding of the method/algorithm, not just the knowledge of the name and how to use some existing code. As much as possible, avoid statements like ‘we used XYZ to learn a representation’ - briefly (say, in a couple sentences) explain what XYZ does, and add references.

**Experimental Setup:** Clearly describe the experimental setup, e.g., how was training done, architecture choices, performance evaluation measures, did you use cross-validation, sampling, etc., how were hyperparameters tuned,etc.

This is a key component of the final report.  If your project is an argument from literature, explain how you gathered background references and your argumentation strategy.

**Results:** Discuss the results from your experiments in terms of what can be concluded. As relevant, discuss the relative strengths and weaknesses of the methods considered. Present results in a compact form, say figures, plots, etc., as appropriate. Reason about the results in light of theory.

**Conclusions:** In addition to a brief conclusion to the project, discuss possible next steps very briefly. You do not have to complete the ‘next steps’ work as part of the class project, but it will illustrate how you are thinking.

**References:** Few key references (books, papers, etc.)

---

## Final Project Rubric

| Criteria | 10 pts / 25 pts / 15 pts / 20 pts / 15 pts / 10 pts / 5 pts Full Points | 8 pts / 20 pts / 12 pts / 15 pts / 12 pts / 8 pts / 4 pts Partial Credit | 6 pts / 15 pts / 10 pts / 10 pts / 10 pts / 5 pts / 3 pts Partial Credit | 4 pts / 10 pts / 5 pts / 5 pts / 5 pts / 2 pts / 2 pts Partial Credit | 0 pts No Points | Pts |
|---|---|---|---|---|---|---|
| **1. Problem Statement & Background**<br><br>Clarity of the problem, motivation, literature review, and connection to course topics. | **10 pts**<br>**Full Points**<br><br>Precisely defined problem with strong motivation, excellent synthesis of 5+ high-quality references, clear identification of gaps, and explicit ties to stochastic AI concepts. | **8 pts**<br>**Partial Credit**<br><br>Well-defined with good literature, solid gap analysis. | **6 pts**<br>**Partial Credit**<br><br>Adequate definition and background but limited depth. | **4 pts**<br>**Partial Credit**<br><br>Vague problem or weak/incomplete background. | **0 pts**<br>**No Points**<br><br>No clearly defined problem statement or cited reference. | **10 pts** |
| **2. Methodology & Technical Approach**<br><br>Soundness, originality, and implementation quality of the proposed solution. | **25 pts**<br>**Full Points**<br><br>Innovative approach; clear justification of design choices; stochasticity is deeply integrated and justified with equations/pseudocode. | **20 pts**<br>**Partial Credit**<br><br>Solid methodology with good technical depth and stochastic elements; code is functional and reasonably organized. | **15 pts**<br>**Partial Credit**<br><br>Basic approach implemented; stochasticity present but superficial. | **10 pts**<br>**Partial Credit**<br><br>Weak methodology, missing key components, no/little stochasticity. | **0 pts**<br>**No Points**<br><br>Selected methodology does not handle stochastic elements of the problem. | **25 pts** |
| **3. Implementation & Reproducibility**<br><br>Completeness of code, data handling, and ability for others to reproduce results. | **15 pts**<br>**Full Points**<br><br>Fully implemented, well-documented codebase including README, requirements, scripts; data preprocessing clear; results fully reproducible; all proposal elements delivered or justified. | **12 pts**<br>**Partial Credit**<br><br>Mostly complete and reproducible with minor gaps. | **10 pts**<br>**Partial Credit**<br><br>Functional but documentation or reproducibility issues. | **5 pts**<br>**Partial Credit**<br><br>Incomplete implementation or hard to reproduce. | **0 pts**<br>**No Points**<br><br>No code submitted or linked on GitHub. | **15 pts** |
| **4. Experiments & Results**<br><br>Quality of experimental design, execution, and presentation of outcomes. | **20 pts**<br>**Full Points**<br><br>Rigorous design with multiple runs to account for stochasticity; clear tables/figures; baselines, and hyperparameter exploration; results visually well-presented. | **15 pts**<br>**Partial Credit**<br><br>Good experiments with some baselines; stochastic variability addressed. | **10 pts**<br>**Partial Credit**<br><br>Adequate experiments but limited scope or analysis of randomness. | **5 pts**<br>**Partial Credit**<br><br>Insufficient experiments or poor presentation. | **0 pts**<br>**No Points**<br><br>No mention of experiments or results. | **20 pts** |
| **5. Evaluation & Analysis**<br><br>Appropriate metrics, statistical handling of stochasticity, and insightful interpretation. | **15 pts**<br>**Full Points**<br><br>Strong metrics including uncertainty measures; statistical tests such confidence intervals, significance; thorough analysis of stochastic behavior, limitations, and failure cases; meaningful insights. | **12 pts**<br>**Partial Credit**<br><br>Solid metrics and basic statistical analysis of results. | **10 pts**<br>**Partial Credit**<br><br>Basic evaluation but weak statistical handling of randomness. | **5 pts**<br>**Partial Credit**<br><br>Inappropriate metrics or minimal analysis. | **0 pts**<br>**No Points**<br><br>No analysis or evaluation metrics. | **15 pts** |
| **6. Discussion, Conclusions & Future Work**<br><br>Interpretation of results, limitations, ethical considerations, and extensions. | **10 pts**<br>**Full Points**<br><br>Deep discussion of findings versus literature; honest discussion of limitations (esp. stochastic aspects); ethical implications addressed if applicable; thoughtful future work mentioned. | **8 pts**<br>**Partial Credit**<br><br>Good discussion and limitations noted. | **5 pts**<br>**Partial Credit**<br><br>Basic conclusions with limited depth. | **2 pts**<br>**Partial Credit**<br><br>Superficial discussion. | **0 pts**<br>**No Points**<br><br>No conclusive remarks. | **10 pts** |
| **7. Report Quality & Professionalism**<br><br>Clarity, organization, writing, citations, and formatting. | **5 pts**<br>**Full Points**<br><br>Excellent academic writing; logical flow; professional formatting using LaTeX; error-free; all figures/tables captioned and referenced; proper citations. | **4 pts**<br>**Partial Credit**<br><br>Clear and well-organized with minor issues. | **3 pts**<br>**Partial Credit**<br><br>Readable but noticeable writing/organization problems. | **2 pts**<br>**Partial Credit**<br><br>Poor writing, structure, or formatting. | **0 pts**<br>**No Points**<br><br>Report not generated using Latex. | **5 pts** |

**Total Points: 100**

---

## Required components

- **Required components:** Regardless of form the write up must include:
  - Formal problem statement
    - Clearly express assumptions about domain, agents, communication, representations, domains, etc.
    - Articulate goals of agent in logical/mathematical form.
    - Agents MUST have a MODEL of their OWN UNCERTAINTY, which means that NO MODEL-FREE RL PROJECTS ARE ADMISSIBLE.
  - Justification/significance of the problem AND of the approach taken by the project
  - Results: Develop theory, (and/or) test domain, (and/or) implemention. Achieving slightly better performance on an existing testbed will not be considered as a project relevant result.
  - Directed primary literature review of related work, foundational background and problem domain.
    - This survey counts for 30% of your project grade and should show your ability to independently find, read, understand, and summarize papers in the primary literature related to your project topic.

---

## Additional stochasticity requirement

must demonstrate a clear, feasible plan for a final project that explicitly incorporates stochasticity (e.g., probabilistic models, stochastic optimization, Monte Carlo methods, stochastic processes, reinforcement learning with randomness, Bayesian inference, noisy/uncertain environments, etc.). Purely deterministic approaches will receive 0 points in the stochasticity sub-criterion.

---

# Prompt to Analyze the Current Codebase, Finish Requirements According to the Rubric, and Complete the Final Report

You are an expert AI systems engineer, stochastic AI researcher, technical writer, and LaTeX report editor. I will give you access to my current project repository/codebase, current documentation, README files, existing report drafts, screenshots, experiment outputs, and any related notes. Your job is to deeply analyze the entire codebase and documentation, identify every missing requirement according to the final project outline and rubric below, finish or specify the necessary implementation/experiment/report tasks, and then produce a complete final report with screenshots, visualizations, captions, citations, and reproducibility details.

Do not treat this as a generic writing task. First understand the actual implementation from the repository. Read the code, folder structure, configuration files, scripts, README, requirements, notebooks, tests, data files, logs, docs, and any existing report material. Then map what exists to the rubric and fill the gaps. The final outcome must be a report that directly maximizes the rubric score, especially the stochasticity, uncertainty-modeling, experimental design, evaluation, reproducibility, and literature-review criteria.

## Source Requirements to Preserve

Use the following project requirements and grading expectations exactly as constraints while analyzing and completing the work:

### Suggested outline for the final report

**Suggested outline for the final report.**

**Introduction:** Introduction of the problem and the data/environment you are using.

Introduction should include:

- Problem introduction:  Significance of the problem (why it's important to solve it for intellectual and/or social, economic reasons etc.).
- Problems should be stated in terms of what needs to be solved by the agent.
- What is assumed known or given as background to the agent or problem and how the agent.
- What background is there to the problem - contextualize the problem in terms of relevant literature.
- Why is this problem a "problem" - what characteristics make it challenging or fruitful. Many AI problems are smaller learning/inference/optimization problems embedded in an overall agent's design- make sure you describe the overall problem and the key subproblem of interest.
- How is the data or environment appropriate?  How will the agent interact with your simulation or environment?
- Sketch your overall solution strategy.

**Methods and Algorithms:**

Clearly describe the methods and algorithms used in the projects including any feature engineering, any (external) tools or pre-trained models you used. By ‘clearly’ I mean illustrate your understanding of the method/algorithm, not just the knowledge of the name and how to use some existing code. As much as possible, avoid statements like ‘we used XYZ to learn a representation’ - briefly (say, in a couple sentences) explain what XYZ does, and add references.

**Experimental Setup:** Clearly describe the experimental setup, e.g., how was training done, architecture choices, performance evaluation measures, did you use cross-validation, sampling, etc., how were hyperparameters tuned,etc.

This is a key component of the final report.  If your project is an argument from literature, explain how you gathered background references and your argumentation strategy.

**Results:** Discuss the results from your experiments in terms of what can be concluded. As relevant, discuss the relative strengths and weaknesses of the methods considered. Present results in a compact form, say figures, plots, etc., as appropriate. Reason about the results in light of theory.

**Conclusions:** In addition to a brief conclusion to the project, discuss possible next steps very briefly. You do not have to complete the ‘next steps’ work as part of the class project, but it will illustrate how you are thinking.

**References:** Few key references (books, papers, etc.)

### Final Project Rubric

| Criteria | 10 pts / 25 pts / 15 pts / 20 pts / 15 pts / 10 pts / 5 pts Full Points | 8 pts / 20 pts / 12 pts / 15 pts / 12 pts / 8 pts / 4 pts Partial Credit | 6 pts / 15 pts / 10 pts / 10 pts / 10 pts / 5 pts / 3 pts Partial Credit | 4 pts / 10 pts / 5 pts / 5 pts / 5 pts / 2 pts / 2 pts Partial Credit | 0 pts No Points | Pts |
|---|---|---|---|---|---|---|
| **1. Problem Statement & Background**<br><br>Clarity of the problem, motivation, literature review, and connection to course topics. | **10 pts**<br>**Full Points**<br><br>Precisely defined problem with strong motivation, excellent synthesis of 5+ high-quality references, clear identification of gaps, and explicit ties to stochastic AI concepts. | **8 pts**<br>**Partial Credit**<br><br>Well-defined with good literature, solid gap analysis. | **6 pts**<br>**Partial Credit**<br><br>Adequate definition and background but limited depth. | **4 pts**<br>**Partial Credit**<br><br>Vague problem or weak/incomplete background. | **0 pts**<br>**No Points**<br><br>No clearly defined problem statement or cited reference. | **10 pts** |
| **2. Methodology & Technical Approach**<br><br>Soundness, originality, and implementation quality of the proposed solution. | **25 pts**<br>**Full Points**<br><br>Innovative approach; clear justification of design choices; stochasticity is deeply integrated and justified with equations/pseudocode. | **20 pts**<br>**Partial Credit**<br><br>Solid methodology with good technical depth and stochastic elements; code is functional and reasonably organized. | **15 pts**<br>**Partial Credit**<br><br>Basic approach implemented; stochasticity present but superficial. | **10 pts**<br>**Partial Credit**<br><br>Weak methodology, missing key components, no/little stochasticity. | **0 pts**<br>**No Points**<br><br>Selected methodology does not handle stochastic elements of the problem. | **25 pts** |
| **3. Implementation & Reproducibility**<br><br>Completeness of code, data handling, and ability for others to reproduce results. | **15 pts**<br>**Full Points**<br><br>Fully implemented, well-documented codebase including README, requirements, scripts; data preprocessing clear; results fully reproducible; all proposal elements delivered or justified. | **12 pts**<br>**Partial Credit**<br><br>Mostly complete and reproducible with minor gaps. | **10 pts**<br>**Partial Credit**<br><br>Functional but documentation or reproducibility issues. | **5 pts**<br>**Partial Credit**<br><br>Incomplete implementation or hard to reproduce. | **0 pts**<br>**No Points**<br><br>No code submitted or linked on GitHub. | **15 pts** |
| **4. Experiments & Results**<br><br>Quality of experimental design, execution, and presentation of outcomes. | **20 pts**<br>**Full Points**<br><br>Rigorous design with multiple runs to account for stochasticity; clear tables/figures; baselines, and hyperparameter exploration; results visually well-presented. | **15 pts**<br>**Partial Credit**<br><br>Good experiments with some baselines; stochastic variability addressed. | **10 pts**<br>**Partial Credit**<br><br>Adequate experiments but limited scope or analysis of randomness. | **5 pts**<br>**Partial Credit**<br><br>Insufficient experiments or poor presentation. | **0 pts**<br>**No Points**<br><br>No mention of experiments or results. | **20 pts** |
| **5. Evaluation & Analysis**<br><br>Appropriate metrics, statistical handling of stochasticity, and insightful interpretation. | **15 pts**<br>**Full Points**<br><br>Strong metrics including uncertainty measures; statistical tests such confidence intervals, significance; thorough analysis of stochastic behavior, limitations, and failure cases; meaningful insights. | **12 pts**<br>**Partial Credit**<br><br>Solid metrics and basic statistical analysis of results. | **10 pts**<br>**Partial Credit**<br><br>Basic evaluation but weak statistical handling of randomness. | **5 pts**<br>**Partial Credit**<br><br>Inappropriate metrics or minimal analysis. | **0 pts**<br>**No Points**<br><br>No analysis or evaluation metrics. | **15 pts** |
| **6. Discussion, Conclusions & Future Work**<br><br>Interpretation of results, limitations, ethical considerations, and extensions. | **10 pts**<br>**Full Points**<br><br>Deep discussion of findings versus literature; honest discussion of limitations (esp. stochastic aspects); ethical implications addressed if applicable; thoughtful future work mentioned. | **8 pts**<br>**Partial Credit**<br><br>Good discussion and limitations noted. | **5 pts**<br>**Partial Credit**<br><br>Basic conclusions with limited depth. | **2 pts**<br>**Partial Credit**<br><br>Superficial discussion. | **0 pts**<br>**No Points**<br><br>No conclusive remarks. | **10 pts** |
| **7. Report Quality & Professionalism**<br><br>Clarity, organization, writing, citations, and formatting. | **5 pts**<br>**Full Points**<br><br>Excellent academic writing; logical flow; professional formatting using LaTeX; error-free; all figures/tables captioned and referenced; proper citations. | **4 pts**<br>**Partial Credit**<br><br>Clear and well-organized with minor issues. | **3 pts**<br>**Partial Credit**<br><br>Readable but noticeable writing/organization problems. | **2 pts**<br>**Partial Credit**<br><br>Poor writing, structure, or formatting. | **0 pts**<br>**No Points**<br><br>Report not generated using Latex. | **5 pts** |

**Total Points: 100**

### Required components

- **Required components:** Regardless of form the write up must include:
  - Formal problem statement
    - Clearly express assumptions about domain, agents, communication, representations, domains, etc.
    - Articulate goals of agent in logical/mathematical form.
    - Agents MUST have a MODEL of their OWN UNCERTAINTY, which means that NO MODEL-FREE RL PROJECTS ARE ADMISSIBLE.
  - Justification/significance of the problem AND of the approach taken by the project
  - Results: Develop theory, (and/or) test domain, (and/or) implemention. Achieving slightly better performance on an existing testbed will not be considered as a project relevant result.
  - Directed primary literature review of related work, foundational background and problem domain.
    - This survey counts for 30% of your project grade and should show your ability to independently find, read, understand, and summarize papers in the primary literature related to your project topic.

### Additional stochasticity requirement

must demonstrate a clear, feasible plan for a final project that explicitly incorporates stochasticity (e.g., probabilistic models, stochastic optimization, Monte Carlo methods, stochastic processes, reinforcement learning with randomness, Bayesian inference, noisy/uncertain environments, etc.). Purely deterministic approaches will receive 0 points in the stochasticity sub-criterion.

---

## Phase 0: Repository and Documentation Intake

1. Inspect the complete repository tree before writing anything.
2. Read all README files, architecture docs, proposal docs, notebooks, scripts, package files, configuration files, API definitions, model code, data-processing code, experiment code, tests, and existing report drafts.
3. Identify the actual implemented system, not just the intended system.
4. Record the exact commands needed to install dependencies, run the application, run experiments, regenerate figures, and reproduce results.
5. Identify where screenshots should be captured from the application and what each screenshot should prove for the report.
6. Do not invent implementation details. If something is missing, mark it as a gap and either implement it or write a concrete TODO with file paths, expected code changes, commands, and acceptance checks.

Deliverable for this phase:

- A concise codebase map with important files and modules.
- A list of implemented features.
- A list of missing or weak features relative to the rubric.
- A reproducibility command checklist.

## Phase 1: Rubric-to-Codebase Gap Audit

Create a rubric coverage matrix with the following columns:

- Rubric criterion
- Required evidence in the report
- Evidence currently present in the codebase/docs
- Missing evidence or weak spots
- Exact action needed
- Priority level
- File(s) to modify or create
- How to verify completion

The audit must explicitly check for:

1. A precise problem statement.
2. Clear motivation and significance.
3. At least 5 high-quality references, preferably primary literature.
4. Clear literature synthesis and gap analysis.
5. Explicit connection to stochastic AI concepts.
6. A formal agent/environment/problem formulation.
7. Assumptions about domain, agents, communication, representations, and environment.
8. Goals of the agent in logical/mathematical form.
9. A model of the agent's own uncertainty.
10. Non-deterministic/stochastic elements integrated into the methodology.
11. Equations or pseudocode showing stochasticity/uncertainty handling.
12. Reproducible implementation with README, requirements, scripts, and data-processing instructions.
13. Experiments with multiple runs or seeds.
14. Baselines and ablations where feasible.
15. Hyperparameter exploration where feasible.
16. Metrics that include uncertainty measures.
17. Statistical analysis such as confidence intervals, variance, standard deviation, significance tests, or other uncertainty-aware analysis.
18. Failure cases and limitations.
19. Ethical considerations if applicable.
20. Screenshots, figures, tables, captions, and references in LaTeX.

Deliverable for this phase:

- A rubric coverage matrix.
- A prioritized execution plan.
- A list of exact artifacts required for the final report.

## Phase 2: Formal Problem and Stochastic AI Formulation

Based on the codebase, formalize the project as an AI agent problem. The formulation must include:

1. Agent definition.
2. Environment or test domain definition.
3. State representation.
4. Observation representation.
5. Action space.
6. Transition dynamics, including stochastic or uncertain components.
7. Reward, objective, utility, or goal function if applicable.
8. Agent uncertainty model.
9. Belief state, probability distribution, Bayesian component, Monte Carlo component, noisy observation model, stochastic process, stochastic optimization, or other explicit stochastic mechanism.
10. How the agent updates, uses, or reasons about its uncertainty.
11. Why this stochasticity is not superficial and why a deterministic approach would be weaker.

Use equations and pseudocode where appropriate. The final report must make it impossible for the grader to say that the project is purely deterministic or lacks a model of its own uncertainty.

Deliverable for this phase:

- Formal problem statement.
- Mathematical/logical agent goal.
- Stochasticity and uncertainty-modeling section.
- Pseudocode or equations for the uncertainty-aware component.

## Phase 3: Literature Review and Background Completion

Perform a directed primary literature review related to the project topic and the stochastic/uncertainty-based method used. Prioritize primary papers, foundational books, and reputable technical references. The literature review must not be a generic list. It must synthesize the papers and explain how they motivate the project design.

The literature review must include:

1. At least 5 high-quality references.
2. A short explanation of each reference's relevance.
3. A synthesis of common themes.
4. A clear gap that this project addresses.
5. Connection to stochastic AI, uncertainty modeling, probabilistic reasoning, stochastic optimization, Monte Carlo methods, Bayesian inference, stochastic processes, noisy environments, or reinforcement learning with randomness, depending on what fits the codebase.
6. Proper citations in the final LaTeX report.

Deliverable for this phase:

- A literature review subsection.
- A related work table if useful.
- BibTeX entries or a references section.

## Phase 4: Implementation Completion and Reproducibility

After identifying gaps, update or create the necessary implementation and documentation artifacts. Do not over-engineer. Focus on changes that directly improve the rubric score.

Check and complete:

1. README with setup, run, experiment, and reproduction instructions.
2. requirements.txt, environment.yml, package.json, pyproject.toml, or equivalent dependency files.
3. Scripts for running experiments.
4. Scripts for generating plots/tables.
5. Data preprocessing steps and sample data instructions.
6. Seed control and multiple-run support.
7. Logging of stochastic outputs, uncertainty measures, and experiment results.
8. Clear folder organization for results, figures, screenshots, and report assets.
9. Comments/docstrings for key methods.
10. Any missing proposal elements, or a justification for why they were not delivered.

Deliverable for this phase:

- Updated code/docs or a patch plan.
- Reproducibility instructions.
- Verification commands and expected outputs.

## Phase 5: Experiment Design and Execution

Design and run experiments that satisfy the rubric rather than only demonstrating that the app works.

The experiments should include, where feasible:

1. Multiple runs with different random seeds.
2. At least one baseline comparison.
3. At least one ablation or variant comparison.
4. Hyperparameter exploration if relevant.
5. Metrics that capture both performance and uncertainty/stochastic behavior.
6. Tables summarizing mean, standard deviation, confidence intervals, and number of runs.
7. Visualizations such as line plots, bar charts, histograms, confidence interval plots, or failure-case examples.
8. Qualitative screenshots from the running application.
9. Failure cases and error analysis.

If experiments cannot be fully run in the current environment, create the experiment scripts and clearly mark which commands must be run locally, what outputs they will generate, and where those outputs should be inserted into the report.

Deliverable for this phase:

- Experiment plan.
- Experiment scripts or commands.
- Result tables.
- Generated visualizations.
- Interpretation notes.

## Phase 6: Screenshots and Visualizations

The final report must include screenshots and visualizations with captions and references in the text. Analyze the application and identify the most useful screenshots.

Include screenshots such as:

1. Main application UI showing the agent/environment interaction.
2. A screen showing the agent taking or recommending an action.
3. A screen showing stochastic/uncertainty-related behavior, such as uncertainty estimates, probabilistic sampling, memory retrieval variability, confidence scores, noisy observations, or multiple sampled outcomes.
4. A debug, log, or telemetry view if available.
5. Experiment output screens or generated plots.
6. Architecture diagram or pipeline diagram.

For every screenshot/figure, provide:

- File name.
- Where/how to capture it.
- What rubric criterion it supports.
- Caption.
- Where it should be referenced in the report.

Deliverable for this phase:

- Screenshot checklist.
- Figure list.
- Captions.
- Visualization generation commands.

## Phase 7: Final Report Construction in LaTeX

Write the final report in LaTeX. The report must be academically clear but still human-sounding. Avoid vague claims. Every figure and table must be captioned and referenced in the text. Every non-obvious method or external model/tool must be explained, not just named.

The report should include:

1. Title and abstract if appropriate.
2. Introduction.
3. Problem statement and motivation.
4. Background and related work / directed primary literature review.
5. Formal problem formulation.
6. Methods and algorithms.
7. Stochasticity and uncertainty model.
8. System architecture / implementation.
9. Experimental setup.
10. Results.
11. Evaluation and analysis.
12. Discussion, limitations, ethics, and future work.
13. Conclusion.
14. References.
15. Appendix if useful for commands, pseudocode, or extra tables.

The report must explicitly include:

- The formal problem statement.
- Assumptions about domain, agents, communication, representations, and environment.
- Agent goals in logical/mathematical form.
- The agent's model of its own uncertainty.
- Stochasticity integrated into the approach and justified with equations/pseudocode.
- Directed primary literature review.
- Results that are more meaningful than slightly better performance on an existing testbed.
- Experiments with multiple runs or stochastic variability handling.
- Statistical analysis and uncertainty measures.
- Discussion of limitations, especially stochastic aspects.
- Ethical implications if applicable.
- Proper citations.
- LaTeX formatting.

Deliverable for this phase:

- Complete .tex report.
- .bib file or references section.
- Figure/table assets.
- List of all unresolved placeholders, if any.

## Phase 8: Final Rubric Self-Check

Before finalizing, grade the report against the rubric. Be strict. For each criterion, estimate the likely score and explain what evidence supports that score.

Create a final checklist:

1. Problem Statement & Background: likely score and evidence.
2. Methodology & Technical Approach: likely score and evidence.
3. Implementation & Reproducibility: likely score and evidence.
4. Experiments & Results: likely score and evidence.
5. Evaluation & Analysis: likely score and evidence.
6. Discussion, Conclusions & Future Work: likely score and evidence.
7. Report Quality & Professionalism: likely score and evidence.

Then identify the top 5 changes that would most improve the grade if more time is available.

Deliverable for this phase:

- Rubric self-check table.
- Final missing-risk list.
- Top 5 grade-improving changes.

## Phase 9: Execution Rules

Follow these execution rules:

1. Do not write a final report before analyzing the actual codebase.
2. Do not invent code behavior that is not present.
3. If implementation is missing, either implement it or provide exact code-level steps to implement it.
4. Do not leave stochasticity vague. It must be explicit, central, and tied to the agent's uncertainty model.
5. Do not rely on one run only if the rubric asks for stochastic variability handling.
6. Do not include figures without captions and in-text references.
7. Do not include references without explaining how they connect to the project.
8. Do not claim reproducibility unless commands and dependencies are documented.
9. Do not skip ethical implications if the project interacts with people, personal data, decision-making, or deployed AI agents.
10. Do not over-polish into generic AI writing. Keep the report clear, specific, technical, and grounded in the repository.

## Final Output Required

Return the following final outputs:

1. Codebase analysis summary.
2. Rubric gap matrix.
3. Phase-wise execution plan.
4. Any code/documentation changes made or required.
5. Experiment plan and results/visualization plan.
6. Screenshot checklist with captions.
7. Complete LaTeX final report.
8. References/BibTeX.
9. Reproducibility instructions.
10. Final strict rubric self-check.
