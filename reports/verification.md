# Verification

Verified in this workspace on 7 October 2026.

| Check | Result |
|---|---|
| Python behavior suite, `python -m pytest -q` | **38 passed** in 2.23 seconds |
| JavaScript syntax, `node --check web/app.js` | Passed |
| Python module compilation | Passed |
| Notebook schema and execution | Validated; all code cells executed on CPU in a temporary output directory |
| Desktop Chromium interaction checks | Passed at 1440 × 1100 |
| Mobile Chromium layout checks | Passed at 390 × 844, with no horizontal overflow |
| Default sample highlights | Two stable, three time-sensitive sentences |
| Historical fee example | Stable; present fee is time-sensitive |
| Required six user examples | All expected labels |
| False stable statement | Classified stable, independently of truth |
| Version and decimal boundaries | Preserved |
| Unicode offsets | Emoji-containing answer highlighted with correct source spans |
| HTML text escaping | Pasted HTML displayed as text; no inserted image element |
| Copy labels | Both label classes copied to clipboard |
| JSON export | Downloaded JSON contained the actual answer and predictions |
| Empty-input feedback | Displayed |
| Keyboard shortcut and example buttons | Passed |
| Browser JavaScript errors | None |
| Automatic browser external requests | None |
| Inference with socket connections blocked | Passed |
| Exact normalized text and source/family group leakage | Zero overlap across each pair of splits |

Preview images: [desktop](app-desktop.png) and [mobile](app-mobile.png). Test accuracy is reported separately in [evaluation.md](evaluation.md). Regression checks of training examples establish app behavior; they are not generalization measurements. Browser checks verify the interface, not model accuracy.
