# Machine Learning End-Sem Prep · MJPRU CS-401T

Study site for **MJPRU Bareilly · B.Tech VII Semester · Machine Learning (CS-401T)**.

**[Open the site](https://hira04irshad.github.io/ml-mjpru-prep/)**

## What's inside

| Page | What it gives you |
|---|---|
| **PYQ analysis** | The 2025-26 end-sem, Mid-2 2025 and an earlier mid paper ranked by topic, with a predicted paper |
| **All past papers** | Every question typed out and tagged by unit, with the k-NN numerical solved |
| **How to write answers** | The MJPRU long-answer template, "differentiate" tables and the layout for numericals |
| **Video lectures** | YouTube lectures for every syllabus topic, unit by unit; play on the page with speed control |
| **Unit notes 1–5** | Bayes decision theory, linear machines and neural networks, decision trees and k-NN, learning theory and ensembles, SVM, with every numerical worked |
| **Mock papers 1–3** | 70-mark papers in the MJPRU pattern with full answers; every numerical checked by computer |
| **Last-night sheet** | Definitions, lists, formulas, differences and the answers to check against |

## How it works

A static site with no build step, using the same design as [cyber-mjpru-prep](https://github.com/HIRA04IRSHAD/cyber-mjpru-prep) and [crypto-mjpru-prep](https://github.com/HIRA04IRSHAD/crypto-mjpru-prep). `index.html` holds the shell and the home page; every other page is `content/<page>.md`, rendered with [marked](https://marked.js.org/) and [Mermaid](https://mermaid.js.org/). Progress and exam mode are saved in the browser only.

To edit content, change the Markdown in `content/` and push to `main`; GitHub Actions publishes to GitHub Pages. Bump the `?v=` number in `index.html` when you change `assets/`. The end-sem date is `EXAM_DATE` in `assets/app.js` (a tentative 15 Dec 2026).

## Sources

Duda, Hart and Stork, *Pattern Classification*; Mitchell, *Machine Learning*; Shalev-Shwartz and Ben-David, *Understanding Machine Learning*; Bishop, *Pattern Recognition and Machine Learning*; and the class Mid-1 notes. The papers are from our batch; no older MJPRU CS-401T papers could be found online.

Not an official MJPRU resource. If anything here disagrees with your class notes, trust your teacher.
