# Energy-Guided Compact SafeSplit

This repository contains our project implementation for reproducing and extending **SafeSplit**, a defense against client-side backdoor attacks in Split Learning.

The project is based on the paper:

> **SafeSplit: A Novel Defense Against Client-Side Backdoor Attacks in Split Learning**

Our goal is to first reproduce the SafeSplit-style defense pipeline, and then propose an efficiency-oriented extension called:

> **Energy-Guided Compact SafeSplit**

The extension reduces the defense-side overhead by analyzing only a compact, automatically selected subset of server-side backbone updates, while keeping the original attack setting, training pipeline, SafeSplit decision rule, and rollback mechanism unchanged.

---

## 1. Project Overview

Split Learning allows multiple clients and a central server to collaboratively train a neural network without sharing raw client data. In the U-shaped Split Learning setting, the model is divided into three parts:

```text
Client-side Head  ->  Server-side Backbone  ->  Client-side Tail
