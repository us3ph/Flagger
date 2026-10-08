# Verification

Verified in this workspace on 8 October 2026. Browser checks used the real FastAPI application through an in-process TestClient and intercepted Chromium requests. No application server was started or restarted.

| Check | Result |
|---|---|
| Python suite, `python -m pytest -q` | **60 passed** in 13.59 seconds after simplifying the UI; all 56 original tests preserved |
| JavaScript syntax | `node --check web/app.js` passed |
| Python compilation | `python -m compileall -q flagger scripts tests` passed |
| Updated notebook | nbformat schema validated; all code cells compiled; full semantic training executed separately through the same training function |
| Classifier upgrade | Fixed pretrained MiniLM sentence vectors combined with training-fitted TF-IDF and a supervised temporal head |
| Existing inference contract | Label mapping, threshold decisions, sentence splitting, offsets and count-only summaries preserved |
| Cold semantic load with network blocked | Local encoder loaded and forecast/history predictions completed |
| Movable model bundle | Copied artifact and hardlinked encoder assets load and predict from a different directory |
| Missing encoder | Clear training-command error; no automatic download |
| Saved provenance | 3,223 training rows and recorded SHA-256 match the current CSV |
| Frozen validation/tests | Original validation and both test CSVs unchanged; exact text and source/family group leakage checks pass |
| Paraphrase diagnostic | 60 texts do not exactly overlap any existing split; excluded from fitting and model selection |
| Model replacement | Complete replacement updates metadata and predictions within the same application lifespan |
| Failed or interrupted replacement | Invalid model returns HTTP 503 and valid replacement recovers; interrupted writes preserve the old artifact |
| Browser caching | HTML, JavaScript and model responses use `Cache-Control: no-store` |

| Browser workflow | Result |
|---|---|
| Reference layout | Black background, custom vector FLAGGER wordmark and matching flag logo, compact editor, word count, Clear and Analyze; Examples removed; results and settings initially collapsed |
| Border refinement | Editor width reduced to 980 px, default desktop text area reduced to 320 px, subtle borders and 10 px corners; matching favicon |
| Model preservation during UI change | Classifier sources and trained weights unchanged; model SHA-256 `1e07ed19e0ef0b4c66fd814e6b95b8c34216f8cd2afb10e3e6d043c23d228c8a` |
| Model status and description | Semantic classifier ready; describes fixed pretraining, 22.88-second feature preparation and 0.80-second head fit separately |
| Reported forecast | “euro will go up” is **1**, score **0.981404**; zero stable, one time-sensitive, “1 claim may need current verification” |
| Varied wording and historical contrast | “The euro is expected to appreciate” receives 1; “The euro was introduced in 1999” receives 0 |
| False stable statement | Cube-shaped Earth assertion receives 0 and the notice explains that stable claims can be incorrect |
| Required summary | 15 stable and 2 time-sensitive predictions show “2 claims may need current verification”; no overall truth or trust verdict |
| Label/count consistency | String labels and stale supplied aggregate counts cannot contradict displayed predictions |
| Copy and JSON export | Labels, answer, counts and verification notice agree |
| Threshold change | Keyboard slider input clears previous results before reanalysis |
| Unicode and HTML | Source spans preserve emoji; pasted HTML renders as text |
| Input and tabs | Word count, Clear, empty-input feedback, Ctrl+Enter analysis and keyboard tab navigation work after removing Examples and its handlers |
| Sample answer | Two stable and three time-sensitive sentences; input starts empty |
| Responsive layout | Latest branding refinement checked at widths 320, 375, 390, 700, 900, 1024, 1440 and 1916; no horizontal overflow |
| Browser script errors / external requests | None |
| Existing server on port 8000 | Reports `hybrid_logistic` with the semantic encoder loaded; no server was started or restarted during UI work |

Preview images: [desktop](app-desktop.png), [mobile](app-mobile.png) and [analysis](app-analysis.png). Accuracy comparisons and remaining errors are recorded in [evaluation.md](evaluation.md) and [errors.csv](errors.csv). Both authored evaluation sets are small and lack independent human annotation. Browser checks establish interface behavior; they do not establish general classifier accuracy.
