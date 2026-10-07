"""
setup.py — One-time setup for the Meeting Assistant (Local GPU edition).

Run this once before starting the app:
    python setup.py

What it does:
  1. Checks that the AMI dataset ZIP is present and extracts it.
  2. Runs build_stageA_prompts_from_ami.py to create the few-shot examples
     that the Stage A extraction prompt uses (stageA/stageA_fewshot.json).
  3. Verifies the prompts/ config files are in place.
  4. Prints a summary and the command to launch the app.

The AMI dataset ZIP (ami_json_for_stageA.zip) must be in the parent folder
(d:\\Downloads\\Inter IIT bootcamp\\) OR in the same folder as this script.
"""

import json
import os
import subprocess
import sys
import zipfile

# Fix Windows console encoding so Unicode chars (checkmarks, arrows) display correctly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

# ------------------------------------------------------------------ locate AMI ZIP

AMI_ZIP_CANDIDATES = [
    os.path.join(PARENT, "ami_json_for_stageA.zip"),
    os.path.join(HERE, "ami_json_for_stageA.zip"),
    os.path.join(HERE, "data", "ami_json_for_stageA.zip"),
]

def find_ami_zip():
    for p in AMI_ZIP_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def find_ami_dir(root):
    """Walk root and find the folder containing summlink/ + dialogueActs/."""
    for dirpath, dirnames, _ in os.walk(root):
        if "summlink" in dirnames and "dialogueActs" in dirnames:
            return dirpath
        # Don't recurse more than 4 levels
        if dirpath.replace(root, "").count(os.sep) >= 4:
            dirnames.clear()
    return None


# ------------------------------------------------------------------ main setup

def main():
    print("=" * 60)
    print("  Meeting Assistant — Setup")
    print("=" * 60)
    print()

    errors = []

    # ---- Step 1: AMI dataset ----
    print("[ 1 / 3 ]  AMI dataset")
    ami_dir = find_ami_dir(HERE)
    if not ami_dir:
        ami_dir = find_ami_dir(os.path.join(HERE, "data"))

    if ami_dir:
        print(f"  ✓ Already extracted: {ami_dir}")
    else:
        ami_zip = find_ami_zip()
        if not ami_zip and os.path.isfile(os.path.join(HERE, "stageA", "stageA_fewshot.json")):
            print("  ✓ Not needed: stageA/stageA_fewshot.json is already in the repository.")
            print("    (Add ami_json_for_stageA.zip only to rebuild the few-shot examples.)")
        elif not ami_zip:
            print("  ✗ ami_json_for_stageA.zip not found!")
            print(f"    Looked in: {[os.path.relpath(p, HERE) for p in AMI_ZIP_CANDIDATES]}")
            print("    The app will still work using only the synthetic few-shot example.")
            errors.append("AMI ZIP not found — using synthetic-only Stage A examples.")
        else:
            print(f"  → Extracting {os.path.basename(ami_zip)} ...")
            extract_to = os.path.join(HERE, "data", "ami_json")
            os.makedirs(extract_to, exist_ok=True)
            with zipfile.ZipFile(ami_zip) as zf:
                zf.extractall(extract_to)
            ami_dir = find_ami_dir(extract_to)
            if ami_dir:
                print(f"  ✓ Extracted to: {ami_dir}")
            else:
                print("  ✗ Extracted but could not find summlink/ + dialogueActs/ inside.")
                errors.append("AMI extraction succeeded but structure not recognised.")
                ami_dir = None

    # ---- Step 2: Build Stage A few-shot examples ----
    print()
    print("[ 2 / 3 ]  Stage A few-shot examples (stageA/stageA_fewshot.json)")
    fewshot_path = os.path.join(HERE, "stageA", "stageA_fewshot.json")
    os.makedirs(os.path.join(HERE, "stageA"), exist_ok=True)

    need_rebuild = True
    if os.path.isfile(fewshot_path):
        try:
            examples = json.load(open(fewshot_path, encoding="utf-8"))
            # Rebuild if examples are too long (old version) or too few
            if examples and max(len(e["lines"]) for e in examples) <= 45 and len(examples) >= 3:
                need_rebuild = False
                print(f"  ✓ Already present ({len(examples)} examples, up to date).")
        except Exception:
            pass

    if need_rebuild:
        if ami_dir:
            print(f"  → Building from AMI data at {ami_dir} ...")
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(HERE, "build_stageA_prompts_from_ami.py"),
                    "--ami-json", ami_dir,
                    "--out-dir", os.path.join(HERE, "stageA"),
                ],
                capture_output=True,
                text=True,
                cwd=HERE,
            )
            if result.returncode == 0:
                examples = json.load(open(fewshot_path, encoding="utf-8"))
                print(f"  ✓ Built {len(examples)} few-shot examples from AMI.")
                if result.stdout:
                    for line in result.stdout.strip().splitlines()[-5:]:
                        print(f"    {line}")
            else:
                print("  ✗ build_stageA_prompts_from_ami.py failed:")
                for line in (result.stderr or result.stdout or "").splitlines()[-10:]:
                    print(f"    {line}")
                errors.append("Stage A prompt build failed — check output above.")
        else:
            # Create a minimal placeholder so stageA_prompt.py doesn't crash
            # (it falls back to SYNTHETIC_EXAMPLE automatically)
            if not os.path.isfile(fewshot_path):
                with open(fewshot_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
            print("  ⚠  No AMI data — using synthetic-only example (stageA_fewshot.json = []).")
            print("     The pipeline still works; Stage A accuracy may be slightly lower.")

    # ---- Step 3: Config files ----
    print()
    print("[ 3 / 3 ]  Config files")
    configs = {
        os.path.join(HERE, "prompts", "refine_config.json"):
            '{"use_demo": true, "batch_size": 20}',
        os.path.join(HERE, "prompts", "stageA_config.json"):
            '{"variant": "ami+synthetic"}',
        os.path.join(HERE, "prompts", "glossary.txt"):
            "Kubernetes\nPostgreSQL\nRedis\nCI/CD\nAPI\nOKR\nsprint\nbacklog\n",
    }
    for path, default in configs.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.isfile(path):
            print(f"  ✓ {os.path.relpath(path, HERE)}")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(default)
            print(f"  ✓ Created {os.path.relpath(path, HERE)} (default)")

    # ---- Summary ----
    print()
    print("=" * 60)
    if errors:
        print("  ⚠  Setup completed with warnings:")
        for w in errors:
            print(f"     • {w}")
    else:
        print("  ✅  Setup complete — everything is ready!")
    print()
    print("  Launch the app:")
    print(f"    cd \"{HERE}\"")
    print("    python app.py")
    print()
    print("  First run downloads Whisper + Qwen weights (~7 GB).")
    print("  Subsequent runs are much faster.")
    print("=" * 60)


if __name__ == "__main__":
    main()
