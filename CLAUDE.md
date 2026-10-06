# Instructions

## Project structure
- The root directory contains the Python codebase (managed via standard git/GitHub).
- `./prediction_intervention_games/` holds the LaTeX sources (`main.tex`, `body.tex`, `images/`); it is a separate Overleaf project, listed in the root `.gitignore`, and not versioned by the root git. Never merge root git logic into it.
- Plots/tables/etc used in the paper should be saved in `prediction_intervention_games/images/`. Otherwise, don't save experiment results there. This should stay as clean as possible on Overleaf.

## LaTeX rules (Overleaf compatibility)
- **Semantic line breaks:** Write exactly one sentence per line in `.tex` files. Never reflow, wrap, or auto-format paragraphs.
- **Scope of edits:** Only edit the sentences or paragraphs directly requested. Do not clean up or rephrase untouched text to avoid git conflicts with human collaborators, unless specifically requested.

## Writing style
- Most important rule: writing should be simple, natural, precise, concise, unemotional, not promotional, professional but human. I want to redundancies, fillers, and fancy terms where none are necessary.
- **No fluff:** Eliminate adjectives like "groundbreaking", "novel", "crucial", "testament", "delve", etc. State the mechanism and result directly.
- **Active and precise:** Use active voice where natural (e.g., "We define...", "Algorithm 1 computes..."). Avoid passive ambiguity.
- **Conciseness:** Prefer short, clear sentences. Remove filler introductory phrases.
