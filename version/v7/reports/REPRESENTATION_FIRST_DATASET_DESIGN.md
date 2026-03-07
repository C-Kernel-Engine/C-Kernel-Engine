# Representation-First Dataset Design

## Purpose

This note captures a reusable mental model for designing tokenization, data, and training stages around representation quality rather than vague advice like "use more data."

The core idea:

- the data should follow the representation you want the model to learn
- the tokenizer, dataset, and task contract should expose the right atoms
- generalization depends on whether those atoms learn clean, reusable dense representations

This applies to:

- SVG
- HTML/CSS
- bash
- C
- Python
- IRs
- config languages
- other structured generation domains

## Core Principle

Do not start with:

- how much data do I have?

Start with:

- what are the atoms of this domain?
- what should each atom mean?
- what interactions should the model learn?
- what data distribution will make those meanings learnable?

More data is only useful if it strengthens the right representations.

## Three Kinds of Atoms

Think across three layers.

### 1. Tokenizer atoms

These are the pieces the tokenizer exposes directly.

Examples:

- `[palette:cool]`
- `linearGradient`
- `stroke-width`
- `for`
- `while`
- `#!/bin/bash`

Question:

- does the tokenizer make important concepts cheap or expensive to represent?

### 2. Data atoms

These are the units the dataset repeatedly teaches.

Examples:

- a centered title block
- a gradient card header
- a `for` loop over an array
- a shell pipeline with `grep | sort | uniq`
- a CSS grid layout

Question:

- what recurring semantic pattern does the data reinforce?

### 3. Task atoms

These are the control concepts the task asks the model to obey.

Examples:

- `[layout:grid]`
- `[style:minimal]`
- "write a function"
- "summarize this"
- "generate a flowchart"

Question:

- is the control interface stable, clear, and learnable?

## The Main Loop

For any domain, use this loop.

### 1. Identify the atoms

Find:

- syntax atoms
- semantic atoms
- control atoms
- structural atoms
- numeric/value atoms

Examples for SVG:

- shape atoms
- palette atoms
- layout atoms
- grouping atoms
- coordinate atoms
- gradient atoms

Examples for code:

- identifiers
- operators
- control flow
- indentation / block boundaries
- API call patterns
- type patterns

### 2. Define what each atom must represent

For each important atom, ask:

- what is its local meaning?
- what other atoms should it interact with?
- what long-range role should it play?
- what output changes should happen when it changes?

Example:

- `[palette:cool]` should influence color families, contrast, gradients, and sometimes layout/style pairings
- `for` in Python should influence indentation, iteration variables, collection access, and downstream block structure

### 3. Identify weak representations

An atom may be weak if:

- it is fragmented too much
- it is blobbed with unrelated context
- it appears too rarely
- it appears with inconsistent meaning
- it lacks contrastive examples
- it lacks useful combination coverage
- the task contract using it is unstable

This is the key move.

Do not ask only:

- is loss improving?

Also ask:

- which atoms still have weak meaning?

### 4. Design data to strengthen those atoms

Good data design means:

- repeated clean usage
- controlled variation
- meaningful combinations
- normalized formatting
- explicit contrast where needed
- enough examples of interactions, not just isolated tokens

This is why "more data" alone is weak advice.

Better questions are:

- more data of which atom?
- more variation in which interaction?
- more invariance for which structure?

### 5. Evaluate for generalization

After training, ask:

- does the model reuse the atom compositionally?
- does it work on held-out combinations?
- does changing one control atom produce the expected behavior change?
- does syntax stay valid while semantics remain controllable?

## Common Representation Failures

### 1. Over-fragmentation

Important concepts are split into too many tiny pieces.

Result:

- weak control
- long dependency chains
- brittle prompting

### 2. Over-blobing

Large chunks swallow controllable details.

Result:

- model learns vague habits
- poor editability
- weak compositional control

### 3. Sparse semantics

The atom exists, but appears too rarely or too inconsistently.

Result:

- embedding row exists
- useful meaning does not form strongly

### 4. Surface-only learning

The model learns syntax or local fluency without learning deeper structure.

Result:

- valid output
- weak intent following
- repetitive behavior

### 5. Muddy task contract

The prompt/control language is unstable or ambiguous.

Result:

- later fine-tuning narrows behavior
- instruction following regresses
- model optimizes surface regularities instead of useful control

## Stage-by-Stage View

### Tokenizer stage

Role:

- defines what concepts are efficiently representable

Question:

- are the important atoms visible and stable?

### Pretrain stage

Role:

- teaches the base language and broad structure priors

Question:

- does the model learn how the output domain works at all?

### Midtrain / bridge stage

Role:

- teaches specialized control semantics and structured interactions

Question:

- do the important control atoms start to mean something useful?

### SFT stage

Role:

- teaches exact task obedience

Question:

- does the model reliably map the control interface to the desired output?

### RL / preference stage

Role:

- tunes policy toward rewarded behavior

Question:

- is it refining a good representation, or trying to patch a weak one?

## What This Means In Practice

### SVG

You should think about:

- prompt atoms like palette/layout/style tags
- structural SVG atoms like shapes, gradients, groups, defs
- numeric geometry atoms
- color-role patterns

Then ask:

- which of those are weak?
- what normalized data would strengthen them?

### Code

You should think about:

- syntax/control-flow atoms
- API usage atoms
- block and scope atoms
- naming and typing patterns
- common semantic program templates

Then ask:

- what examples make those patterns reusable rather than memorized?

## Planning Template

Use this table mentally or literally.

### Atom

- what is the atom?

### Desired meaning

- what should changing this atom do?

### Important interactions

- what other atoms should it combine with?

### Failure mode

- how is it weak today?

### Data intervention

- what data would strengthen it?

### Evaluation

- how will I know it generalized?

## Example

### Atom

- `[layout:grid]`

### Desired meaning

- produce multi-element grid-like spatial organization

### Important interactions

- labels
- cards
- palette choices
- numeric spacing

### Failure mode

- token exists but output geometry remains noisy and inconsistent

### Data intervention

- many clean normalized grid examples
- controlled contrast with stacked/horizontal layouts
- stable coordinate conventions

### Evaluation

- held-out prompts with `[layout:grid]`
- verify visible grid structure and controllable changes

## Final Principle

The data follows the mental model.

If your mental model is weak, you get:

- vague datasets
- vague tokenization
- vague evaluations
- vague failures

If your mental model is strong, you can ask:

- which atoms are weak?
- which invariances are missing?
- which combinations are under-taught?
- what data will strengthen the right dense representations?

That is a much better way to design training than just adding more data and hoping the model figures it out.
