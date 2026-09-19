# ⬢ Hex 9x9 AI - Web Inference (ONNX)

[![Play Live](https://img.shields.io/badge/Play-Live_Demo-success?style=for-the-badge&logo=github)](https://damiankrzyzelewski.github.io/hex_web/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime_Web-blue?style=for-the-badge)](https://onnxruntime.ai/)

This repository contains the **web-based inference engine** for my Hex AI project. You can play Hex (9x9) directly in your browser against Deep Reinforcement Learning models without any server-side computations.

The AI models were trained in Python using **PyTorch** and **Stable Baselines 3** (Maskable PPO) and then exported to ONNX format to run locally on the client's machine using WebAssembly.

🔗 **Want to see the training code?** Check out the main Python repository here: `https://github.com/damiankrzyzelewski/HEX9x9_bot`

## 🎮 Play the Game
**[Click here to play the game in your browser!](https://damiankrzyzelewski.github.io/hex_web/)**

### Features
* **Zero Server Costs:** The entire AI inference runs on your device's CPU/GPU inside the browser using `onnxruntime-web`.
* **Multiple Difficulty Levels:** Play against checkpoints from different stages of the PPO training process. The available models represent specific training milestones:
  * **Early:** 1,000,000 training steps (`ppo_early.onnx`)
  * **Intermediate:** 2,000,000 training steps (`ppo_mid.onnx`)
  * **Advanced:** 3,500,000 training steps (`ppo_late.onnx`)
* **Heuristic Graph Bot:** Play against a non-ML agent that calculates optimal moves using 0-1 BFS graph traversal to find the shortest connecting paths.
* **Full Game Rules:** Supports the official Hex pie rule (SWAP), allowing the second player to steal the opening move to balance the first-mover advantage.

## 🏗️ Architecture & Deployment
1. `MaskablePPO` policy networks (ResNet-based) are extracted and exported to `.onnx` files.
2. The game logic (`HexState`) is ported to JavaScript.
3. The UI is built using vanilla HTML/CSS/JS.
4. Hosted via GitHub Pages.
