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

# Generate assets
python utils/generate_assets.py

# Build APK
flet build apk --project "CatanMind"
```

The APK will be in `build/apk/`.

---

## 📋 Project Structure

```
CatanMind/
├── .github/workflows/
│   └── build_apk.yml      # CI/CD pipeline
├── assets/
│   ├── icon.png           # App icon (auto-generated)
│   └── splash.png         # Splash screen (auto-generated)
├── utils/
│   └── generate_assets.py # Asset generator script
├── app.py                 # Main Flet application
├── models.py              # Data models
├── heuristics.py          # AI scoring
├── resource_tracker.py    # Card counting
├── solver_*.py            # AI solvers
├── requirements.txt       # Python dependencies
└── DEPLOY.md              # This file
```

---

## ⚠️ Troubleshooting

### Build fails with "No files found"

1. Check that `utils/generate_assets.py` ran successfully
2. Ensure `assets/` folder has `icon.png` and `splash.png`

### APK crashes on device

1. Check device has Android 6.0+ (API 23)
2. Enable "Install from unknown sources"
3. Check GitHub Actions logs for errors

### Workflow doesn't trigger

1. Ensure you pushed to `master` or `main` branch
2. Check `.github/workflows/build_apk.yml` exists
3. Go to Actions tab and click "Run workflow" manually
