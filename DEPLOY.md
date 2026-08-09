# CatanMind - Deployment Guide

## 🚀 Automatic APK Build

This project uses **GitHub Actions** to automatically build the Android APK.

### How It Works

1. **Push to GitHub** → Workflow triggers automatically
2. **GitHub Actions** runs on Ubuntu with Python + Flutter
3. **APK is built** using `flet build apk`
4. **Download** the APK from the Actions tab

---

## 📱 Getting Your APK

### Step 1: Push Your Code

```bash
git add .
git commit -m "Update for APK build"
git push origin master
```

### Step 2: Check Build Status

1. Go to your GitHub repository
2. Click the **Actions** tab
3. Find the latest "Build Android APK" workflow run
4. Wait for it to complete (usually 5-10 minutes)

### Step 3: Download APK

1. Click on the completed workflow run
2. Scroll to **Artifacts** section
3. Download **CatanMind-APK**
4. Extract and install on your Android device

---

## 🔧 Manual Build (Local)

If you want to build locally:

### Prerequisites

- Python 3.10+
- Flutter SDK 3.19+
- Android SDK

### Build Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build APK
flet build apk --module-name app --project "CatanMind"
```

`assets/icon.png` and `assets/splash.png` are committed, so there is no asset
generation step to run.

The APK will be in `build/apk/`.

---

## 📋 Project Structure

```
CatanMind/
├── .github/workflows/
│   └── build_apk.yml       # CI/CD pipeline: tests, then the APK
├── assets/
│   ├── icon.png            # App icon
│   └── splash.png          # Splash screen
├── catanmind/
│   ├── board.py            # Geometry and graph: tiles, nodes, edges, ports
│   ├── state.py            # All mutable state + the event log behind undo
│   ├── rules.py            # Legality, longest road, largest army, VP
│   ├── scoring.py          # What a spot is worth
│   ├── advisor.py          # What to do next (setup and normal turns)
│   ├── tracker.py          # What the opponents are probably holding
│   ├── view.py             # Fitting the board to a screen, and hit-testing
│   └── ui.py               # The Flet screen
├── tests/                  # 220 tests, no display required
├── app.py                  # Entry point
├── requirements.txt        # Runtime dependency (flet)
├── requirements-dev.txt    # ...plus pytest
└── DEPLOY.md               # This file
```

Each layer is usable without the one above it, so the engine can be exercised
head­less — which is how the tests run.

---

## ⚠️ Troubleshooting

### Build fails with "No files found"

1. Ensure `assets/` folder has `icon.png` and `splash.png`
2. Check the build log for the `flet build apk` step

### APK crashes on device

1. Check device has Android 6.0+ (API 23)
2. Enable "Install from unknown sources"
3. Check GitHub Actions logs for errors

### Workflow doesn't trigger

1. Ensure you pushed to `master` or `main` branch
2. Check `.github/workflows/build_apk.yml` exists
3. Go to Actions tab and click "Run workflow" manually
