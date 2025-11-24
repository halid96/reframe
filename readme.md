# 🎬 Reframe - Open Source - Community Driven

## ℹ️ About
Reframe is a tool for smartly cropping landscape or square videos (e.g., 16:9) and converting them into vertically-oriented, mobile-friendly videos (e.g., 9:16 Shorts, TikToks, and Reels).

Reframe aims to be the definitive open-source solution for accurate and intelligent auto-cropping/reframing of videos. Our goal is to develop a single, robust script that adapts seamlessly to any scenario—be it interviews, group shots, sports, or dynamic action—without requiring users to manually select specific modes or strategies.

Current tools, including industry standards like DaVinci Resolve Studio and Premiere Pro, often struggle with accuracy or are prohibitively expensive and slow. We believe the power of the open-source community can solve this problem, creating a tool that is both accessible and superior in performance.

## � Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/reframe.git
   cd reframe
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run a test:**
   ```bash
   python run_test.py --reframe_scripts_folder_name v1_reframe --challenges_folder_name rishi_interview
   ```

## 🚀 Pull request for a new reframe script

- 📍 **Location:** Create a folder under `/reframe_scripts` containing your `.py` script and a `description.md` file.
- ⚙️ **Technical Requirements:**
    - **CLI Interface:** Your script must be runnable from the command line and accept the following arguments:
      `python script.py <input_video> --output <output_path> --cleanup`
    - **Models:** Automatically download any required ML models to the `/models` folder. Do not commit models to the repo.
- 🏆 **Performance Goal:** Your script must outperform or match the current best script listed in `/statistics/statistics.md`.
    - 📊 Check `/statistics/statistics.md` to see the current passing rate and which challenges are currently unsolved.
    - ✨ Your script should ideally solve at least one challenge that the current best script fails.
- ✅ **Validation:** We will review your generated videos against the `criteria.md` in each challenge folder.

## 🎯 Pull request for a new challenge

- 📍 **Location:** Create a folder inside `/challenges`.
- 📦 **Contents:**
    - 📹 A short video file.
    - 📝 A `criteria.md` file describing the expected reframing behavior (e.g., "Keep the person in the center").

## 🏃 Running Tests (`run_test.py`)

You can use the `run_test.py` script to validate your reframe script against challenges.

**Command Syntax:**
```bash
python run_test.py --reframe_scripts_folder_name <SCRIPT_FOLDER> --challenges_folder_name <CHALLENGE_FOLDER>
```

**Examples:**
```bash
# Run script 'v1_reframe' against challenge 'rishi_interview'
python run_test.py --reframe_scripts_folder_name v1_reframe --challenges_folder_name rishi_interview

# Run script 'v1_reframe' against ALL challenges
python run_test.py --reframe_scripts_folder_name v1_reframe --challenges_folder_name all
```

**Options:**
- `--preview`: Show a preview window during processing.

## 📂 Folder Structure

- ⚔️ **/challenges**: Contains test video clips and criteria for validation.
- 🛠️ **/reframe_scripts**: Contains reframing scripts.
- 📈 **/statistics**: Tracks statistics, performance metrics, and passing rates.
- 🎞️ **/test_results**: Contains tests results, reframed videos.
- 🧠 **/models**: Directory for ML models.
- 🏃 **run_test.py**: Main script to run tests and validate performance.

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

**What this means for you:**
- ✅ **You can** use this software for commercial purposes.
- ✅ **You can** modify and distribute this software.
- ❗ **However**, if you use this software (or a modified version of it) as part of a service (e.g., a web app, SaaS, or backend), you **must** open-source your entire project under the same AGPL-3.0 license.

This requirement is inherited from the **Ultralytics YOLO** models used in this project.

## 📧 Contact

- **Email**: halidkyazim@gmail.com