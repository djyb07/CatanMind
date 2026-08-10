# Shipping CatanMind

Three ways to run it, easiest first. You do **not** need Android tooling on
your machine for the first one.

---

## 1. Build the APK on GitHub (recommended)

GitHub runs the whole Android toolchain for you and hands back a file. This is
free for public repositories.

**Every time you want a new APK:**

1. Push your work:

   ```bash
   git push origin master
   ```

2. Open your repository on github.com and click the **Actions** tab.
3. Click the newest run of **Build Android APK**. It takes roughly 10–20
   minutes; the tests run first, and if any fail the APK is not built.
4. When the run shows a green tick, scroll to **Artifacts** at the bottom.
5. Download **CatanMind-APK**. It arrives as a `.zip` — unzip it to get the
   `.apk`.

**To build without pushing anything:** Actions tab → *Build Android APK* in
the left sidebar → **Run workflow** → **Run workflow**. Same result.

### Putting it on your phone

1. Send yourself the `.apk` (Drive, Telegram, email, USB cable — anything).
2. Open it on the phone. Android will say the app is from an unknown source;
   allow it for whichever app you opened the file with.
3. Install, and it appears in your launcher like any other app.

This is normal and safe for an app you built yourself. Android only asks
because the file did not come from the Play Store.

---

## 2. Build the APK on your own machine

Only worth setting up if you want to build repeatedly without waiting for CI.
You need three things installed first: **Flutter**, the **Android SDK**, and a
**JDK 17**. `flutter doctor` will tell you what is missing.

```bash
pip install -r requirements.txt
flet build apk
```

The APK lands in `build/apk/`.

---

## 3. Run it without packaging anything

On a computer, as a desktop window:

```bash
python app.py
```

In a browser — useful for testing on a phone over your home wifi, since the
phone can open your computer's address:

```bash
flet run --web --port 8550 app.py
```

---

## What controls the app's identity

All of it lives in `pyproject.toml`, so the build command never changes:

| Setting | Where | Notes |
|---|---|---|
| App name | `[tool.flet] product` | What shows under the icon |
| Application id | `[tool.flet] bundle_id` | **Permanent.** Changing it makes a different app that cannot update the old one |
| Version | `[project] version` | The number people see |
| Build number | `[tool.flet] build_number` | **Increase this on every release** — Android refuses to install a build whose number is not higher than the one already on the phone |
| Icon and splash | `assets/icon.png`, `assets/splash.png` | Regenerate with `python tools/make_icons.py` |

### Replacing the icon with an illustrated one

`tools/make_icons.py` draws the icon from the same palette and tile art as the
board, so it always matches the app. If you want something illustrated
instead, drop a **1024×1024 PNG** at `assets/icon.png` and rebuild — nothing
else needs changing.

Prompt for an AI image generator:

> A premium mobile app icon for a board-game strategy assistant. Three
> interlocking hexagonal tiles arranged in a tight triangular cluster, viewed
> straight on, each tile rendered with visible thickness like a physical game
> piece catching light from the upper left. Top tile: dense evergreen forest.
> Lower left: grey stone mountains with snow-capped peaks. Lower right: golden
> wheat field. Rich painterly texture on each tile, warm and inviting.
> Background: deep navy blue ocean with a very subtle darker hexagon lattice.
> Soft ambient shadow beneath the cluster. Clean, modern, professional,
> centred composition with generous margin, no text, no letters, no border.
> Flat-illustration style with soft dimensional shading, not photorealistic.
> Square 1024×1024.

Two things to keep in mind:

- **Leave a margin.** Android crops launcher icons to a circle or squircle and
  only guarantees the middle two-thirds. Anything closer to the edge than that
  gets cut off on some phones.
- **Check it small.** Look at it at about 1cm across before committing. Detail
  that reads beautifully at full size often turns to mud on a home screen.

---

## Getting it onto Google Play

**There is no completely free route onto Google Play.** A Play Console
developer account costs **$25, once, forever** — not per app and not per year.
On top of that, personal accounts opened since 2023 must run a closed test
with **20 testers for 14 continuous days** before they are allowed to publish
publicly. That is the real cost: the waiting, more than the money.

Since Play is the only store that charges, here are the free ways to get the
app to people, in the order I would actually try them:

**Just send the APK.** For yourself and a few friends this is the whole
answer. No account, no review, no waiting.

**GitHub Releases.** Attach the APK to a release and share the link. Free,
permanent, and gives you a version history for nothing. Good when more than a
handful of people want it.

**Firebase App Distribution.** Free. Testers get an email, install a small
helper app, and receive each new build automatically. This is the closest
thing to a real store experience without paying, and it is the sensible step
if you want feedback from a group.

**Amazon Appstore.** A real public store with a free developer account. Much
smaller audience than Play, but it is a genuine store listing.

**F-Droid.** Free and well respected, but only accepts open-source apps and
builds them from source itself, so it takes real setup work.

My suggestion: send the APK directly while it is just you and friends, move to
Firebase App Distribution if you gather testers, and only pay the $25 when you
actually want strangers to find it.

---

## When a build fails

The Actions run turns red and there is a **build-logs** artifact next to where
the APK would have been. Download it and search for the first line containing
`Error`. The two usual causes:

- **A test failed.** The APK job never starts. Fix the test — this is the
  pipeline doing its job.
- **A dependency is not available for Android.** Anything in
  `[project] dependencies` is shipped to the phone and has to work there.
  Keep that list to what the app genuinely needs at runtime; development-only
  tools belong in `requirements-dev.txt`.
