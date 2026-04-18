# AIS DEV2IL 😈 MLOps: Taxi Rides

Welcome to the MLOps Taxi Rides exercises! You are going to explore real-world MLOps practices — 
data management, model training, experiment tracking, and serving predictions — all using a dataset 
of [New York City taxi rides](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

---

## 🛫 Getting Started

1. Fork [this repository](https://github.com/peterrietzler/ais-dev2il-mlops-taxi-rides-v2), clone your fork and open it in PyCharm.
2. Make sure you are working on the `main` branch.
3. Install dependencies:
   ```bash
   uv sync
   ```

You're ready to go! 🚀

---

## 📦 Exercise 1: Feel the Pain — CSV vs Parquet

Before we talk about why Parquet is great, let's experience what life looks like without it.

In the `example-data` folder you'll find the same dataset in two formats:
- `2025-01-01.taxi-rides.csv`
- `2025-01-01.taxi-rides.parquet`

Create a new Python file called `compare.py` in the root of the repository and work through the steps below together with your pair.

### Step 1: Load the CSV and inspect the schema

```python
import pandas as pd

df_csv = pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
print(df_csv.dtypes)
```

Run it with:

```bash
uv run compare.py
```

🤔 Look at the column types carefully:
- What type are `tpep_pickup_datetime` and `tpep_dropoff_datetime`? Are those really the right types for timestamps?
- What type is `ride_time`? It should be a number — but is it?

Open the CSV file in PyCharm and look at the first row. Notice anything odd about `ride_time`?

Someone put `"unknown"` in that field. Because CSV has no schema, pandas silently loaded the entire
column as `object` (string) — **without any warning**. Your data is now silently broken. 😬

### Step 2: Load the Parquet and inspect the schema

```python
df_pq = pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
print(df_pq.dtypes)
```

✅ The datetime columns have the correct type — no extra code needed. The schema is embedded in the file itself.

And that `"unknown"` value? It could never end up in a Parquet file in the first place.
Parquet enforces the column type on **write** — if the data doesn't match, it fails loudly instead of silently corrupting your dataset.

### Step 3: Compare file sizes

```python
import os

csv_size = os.path.getsize('example-data/2025-01-01.taxi-rides.csv')
pq_size  = os.path.getsize('example-data/2025-01-01.taxi-rides.parquet')

print(f'CSV:     {csv_size / 1024 / 1024:.1f} MB')
print(f'Parquet: {pq_size  / 1024 / 1024:.1f} MB')
print(f'Parquet is {csv_size / pq_size:.1f}x smaller')
```

💡 How much disk space would you save if you stored a full year of taxi data as Parquet instead of CSV?

### Step 4: Time the loads

```python
import time

t = time.time()
pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
csv_time = time.time() - t

t = time.time()
pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
pq_time = time.time() - t

print(f'CSV:     {csv_time:.3f}s')
print(f'Parquet: {pq_time:.3f}s')
```

🗣️ **Discuss with your pair:** You are building an ML pipeline that retrains a model every day.
What problems would these differences cause over time?

### 🚀 Level Up

#### Challenge 1: Chuck Norris Never Needs `pd.to_datetime()`

> *Chuck Norris doesn't parse dates. Dates parse themselves in his presence.*

When you loaded the CSV, the datetime columns came in as plain strings (`object`). Fix them manually to match the Parquet schema:

```python
df_csv = pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
df_csv['tpep_pickup_datetime']  = pd.to_datetime(df_csv['tpep_pickup_datetime'])
df_csv['tpep_dropoff_datetime'] = pd.to_datetime(df_csv['tpep_dropoff_datetime'])
print(df_csv.dtypes)
```

Now try to fix `ride_time` the same way:

```python
df_csv['ride_time'] = pd.to_numeric(df_csv['ride_time'])
print(df_csv.dtypes)
```

💥 It crashes with a `ValueError`. And that's actually the *right* reaction — you want to train your model on 
valid data. A `ride_time` of `"unknown"` is useless for training and silently coercing it to `NaN` would just hide the problem deeper.

But here's the thing: **you should never have had to deal with this in the first place.** Parquet enforces the schema on 
write — `"unknown"` could never have ended up in a float column. The problem is caught at the source, not discovered halfway 
through your training pipeline.

#### Challenge 2: A Year Worth of Taxi Rides

In ML training, you rarely load just one day of data — you load months or even a full year.
Simulate this by loading the same file 365 times in a loop for both formats:

```python
import time
import pandas as pd

t = time.time()
for _ in range(365):
    pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
csv_time = time.time() - t

t = time.time()
for _ in range(365):
    pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
pq_time = time.time() - t

print(f'CSV:     {csv_time:.1f}s')
print(f'Parquet: {pq_time:.2f}s')
print(f'Parquet is {csv_time / pq_time:.0f}x faster')
```

⏱️ Every time you retrain your model, you'd be waiting those extra seconds just reading data.
At scale, this adds up fast.

#### Challenge 3: No Code? No Problem!

Explore the Parquet file without writing a single line of Python:

1. In PyCharm's **Project** view, open `example-data/2025-01-01.taxi-rides.parquet`
2. Click **"Edit in Data Wrangler"**
3. Try some transformations:
   - Filter out rows where `outlier` is `True`
   - Sort by `trip_distance` descending
   - Drop the `ride_time` column
4. Click **"Export"** to generate a Python script from your actions
5. Run the generated script with `uv run <script_name>.py`

> Data Wrangler lets you explore visually and then hands you the Python code for free —
> great for getting started with a new dataset fast.

---

## 🗄️ Exercise 2: Data Management with DVC

You now have Parquet files — great format, right size. But here's the next problem: **how do you share large data files with your team?**

Git is designed for code, not data. Try to `git add` a folder with 30 Parquet files and Git will dutifully track every byte forever, bloating your repository. There has to be a better way.

Enter **DVC** (Data Version Control) — it works alongside Git. Git tracks your code and tiny pointer files. DVC tracks your actual data files in a separate storage. Same workflow, sane repository.

### Step 1: Get the data

The file `example-data/data.zip` is already in your repository. Extract it into a `data` folder in the root of your repository.
If you are using PyCharm, make sure to **not** add the files to Git.

Your folder structure should look like this:
```
data/
  2025-01-01.taxi-rides.parquet
  2025-01-02.taxi-rides.parquet
  ...
```

### Step 2: Create your DagsHub repository

[DagsHub](https://dagshub.com) is a collaboration platform built for data scientists and ML engineers. 
Think of it as GitHub — but with built-in support for large data files, experiment tracking, 
and model registries. For now we'll use it purely as a **DVC remote storage** — a place to store and share our Parquet files.

1. Go to [https://dagshub.com](https://dagshub.com) and sign in or up with your GitHub account
2. Click **"Create Repository"** → **"Connect a repository"** → select your GitHub fork
3. In your new DagsHub repo, go to **Your Settings** (in your profile menu in the upper right corner) 
   → *Tokens** and copy the default access token. You will need it a bit later.

### Step 3: Initialise DVC

```bash
uv run dvc init
uv run dvc config core.autostage true
```

This creates a `.dvc` folder (similar to `.git`). The `autostage` setting tells DVC to automatically stage `.dvc` pointer files in Git when you run `dvc add` — one less thing to remember.

### Step 4: Configure the DVC remote

Run these commands. Don't forget to replace the placeholders with your actual values using your username, repository name and the token
that you copied above.  Alternatively you can go to your DagsHub repository **Remote** → **DVC** and copy the exact commands from there.

```bash
uv run dvc remote add origin s3://dvc
uv run dvc remote modify origin endpointurl https://dagshub.com/<YOUR USERNAME>/<YOUR REPO>.s3
uv run dvc remote modify origin --local access_key_id <YOUR TOKEN>
uv run dvc remote modify origin --local secret_access_key <YOUR TOKEN>
uv run dvc remote default origin
```

Here's what each command does:

- **`dvc remote add origin s3://dvc`** — registers a new DVC remote called `origin` using the S3 protocol.
- **`dvc remote modify origin endpointurl ...`** — tells DVC where the S3 storage actually lives. DagsHub provides an S3-compatible HTTP endpoint for every repository.
- **`dvc remote modify origin --local access_key_id ...`** — sets your DagsHub token as the S3 access key. The `--local` flag stores this in `.dvc/config.local`, which is gitignored and **never committed**.
- **`dvc remote modify origin --local secret_access_key ...`** — same token again, used as the S3 secret key. DagsHub uses one token for both.
- **`dvc remote default origin`** — marks `origin` as the default remote. Without this, you'd have to specify `-r origin` on a lot of commands

### Step 5: Track Parquet Files

```bash
uv run dvc add data/*.parquet
```

DVC creates one `.dvc` pointer file per Parquet file (e.g. `data/2025-01-01.taxi-rides.parquet.dvc`) and adds `data/*.parquet` to `data/.gitignore` automatically.
Look them up in the data folder. Also have a look at the contents of `data/.gitignore`

Add the data folder to Git:

```bash
git add data
```

Run `git status` — Git now tracks all the `.dvc` pointer files and the updated `.gitignore`, but **not** the raw Parquet files. 
Each pointer file is very small and tells DVC where to find the actual data. The Parquet files are still there on your disk, but Git is 
blissfully unaware of them and will not store them.

### Step 6: Check what DVC wants to push

Before pushing, check which files DVC needs to upload to the remote:

```bash
uv run dvc status --cloud
```

You should see all 30 Parquet files listed as `new file` — they exist locally but haven't been pushed yet. 
After `dvc push` in the next step, running this again will show nothing, meaning everything is in sync.

### Step 7: Push the data

```bash
uv run dvc push
```

Your Parquet files are now stored in your DagsHub remote. DVC uploads them directly to the remote storage without going through Git at all.

### Step 8: Commit the pointer files to Git

```bash
git add data/
git commit -m "Track data files with DVC"
git push
```

### Step 9: "Gone but not Forgotten"

Delete all the Parquet files from your `data` folder:

```bash
rm data/*.parquet
```

Run `dvc pull` to restore them:

```bash
uv run dvc pull
```

🎉 The files are back. Your data is safe, versioned and shareable.

> **💡 The DVC Workflow — every time you pull or switch branches**
>
> Whenever you run `git pull` or `git checkout <branch>`, Git updates the `.dvc` pointer files — but **not** the actual data files. You need to follow up with:
>
> ```bash
> git pull          # updates .dvc pointer files
> uv run dvc pull   # downloads the matching data from the remote
> ```
>
> Think of `.dvc` files as Git's way of saying *"the data should look like this"* — and `dvc pull` as the step that makes it actually happen.

### 🚀 Level Up

#### Challenge 1: Gone but not Forgotten (Remote Edition)

On your **pair's machine**, clone your GitHub fork, configure the DVC remote credentials and run:

```bash
uv run dvc pull
```

Does it work? Your pair should now have the exact same data as you — pulled from *your* DagsHub remote. This is how teams share data without emailing zip files around.

#### Challenge 2: Catch the Change

Let's simulate updating your dataset. Create a small Python script that:
1. Loads `data/2025-01-01.taxi-rides.parquet`
2. Filters out all rows where `outlier` is `True`
3. Saves it back to the same file

Now use DVC to detect what changed:

```bash
uv run dvc status
```

You should now see that the file `data/2025-01-01.taxi-rides.parquet` has changed locally. 
DVC detects this because the hash of the file content has changed. In order to get Git and 
the DVC remote back in sync, you need to add the updated file to DVC, push it to the remote, and commit the pointer file to Git.

Then make sure everything lands correctly in both DVC and Git:

```bash
uv run dvc add data/2025-01-01.taxi-rides.parquet
uv run dvc push
git add data/2025-01-01.taxi-rides.parquet.dvc
git commit -m "Update data: filter outliers, add February data"
git push
```

🤔 **What happens if you forget to `dvc push` before `git push`?**
Your teammate runs `dvc pull` and gets an error — the pointer in Git points to 
data that doesn't exist in the remote yet. Always push DVC before Git!

---

## 🥒 Exercise 3: Train the Model — Pickle Rick

> *"Pickle Rick" is a famous episode of the animated series Rick and Morty, in which the main character Rick turns himself into a pickle. 🥒 In German: **Gurken-Rick** — which sounds significantly less cool.*

You have data. Time to train a model. But before we talk about how to save and track it properly,
let's experience what happens when you don't.

### Step 1: Get the training script ready

The repository contains a file called `outlier_detector_training_skeleton.py`. This is your starting point.
Rename it to `outlier_detector_training.py`.

Open it in PyCharm and read through it. The script is short on purpose. It:
- Accepts a model type as a command-line argument (`random_forest`, `random_forest_v2`, or `logistic_regression`)
- Calls the corresponding training function from `model_trainings.py`
- Prints the evaluation results to the console

The actual training logic lives in `model_trainings.py`. You don't need to edit that file — just know
it's there and that it loads a Parquet file, trains a scikit-learn classifier, and returns the
trained model together with a classification report.

### Step 2: Prepare the training data

The training script expects a single combined Parquet file at `data/taxi-rides-training-data.parquet`.
The `combine_taxi_ride_data.py` script creates it by merging the individual daily files from your `data` folder.

You have two options:

**Single day** (recommended for this exercise — trains in seconds):
```bash
uv run combine_taxi_ride_data.py 2025-01-01
```

**Multiple days** (recommended to create different versions of your model — trains in seconds):
```bash
uv run combine_taxi_ride_data.py 2025-01-01 2025-01-02 2025-01-03
```

**All days** (used later in the CI pipeline — takes much longer):
```bash
uv run combine_taxi_ride_data.py
```

⏱️ Stick with a single day for now. The model will be less accurate than one trained on the full
dataset, but it's good enough to explore with and the feedback loop is much faster.

### Step 3: Track the combined data file with DVC

The combined training data file is just as important as the individual daily files — it's what your model
was actually trained on. Track it with DVC so your team can always reproduce the exact same training run:

```bash
uv run dvc add data/taxi-rides-training-data.parquet
uv run dvc push
git add data/taxi-rides-training-data.parquet.dvc
git commit -m "Add combined training data"
git push
```

### Step 4: Run the training — and lose the results

Run the training for the `random_forest` model type:

```bash
uv run outlier_detector_training.py random_forest
```

You'll see evaluation metrics printed to the console — precision, recall, f1-score. 

Now run it again:

```bash
uv run outlier_detector_training.py random_forest
```

The numbers flash by again. And that's it. There's no file. No record. No way to compare the two runs.
You just trained a model twice and have nothing to show for it. 😬

### Step 5: Save the model as a pickle file

[Pickle](https://docs.python.org/3/library/pickle.html) is Python's built-in serialisation format.
It turns any Python object — including a trained scikit-learn model — into a binary file that you can
save to disk and load back later. Think of it as freezing the model in time.

Add the following to `train_model()` in your `outlier_detector_training.py`, right after the
`logger.info("Model training completed")` line:

```python
os.makedirs("models", exist_ok=True)

model_output_file = f"models/{model_type}.pkl"
logger.info(f"Storing model to: {model_output_file}")
with open(model_output_file, "wb") as f:
    pickle.dump(model, f)
```

Run training again. Check the `models` folder — your `.pkl` file is there!

> 💡 `"wb"` means "write binary". Pickle files are binary, not text, so you need this mode.
> `pickle.dump(model, f)` serialises the model object and writes it into the file.

### Step 6: Save the evaluation metadata

The model file tells you *what* was trained. The metadata tells you *how well* it performed.
Save it alongside the model so you always have both:

```python
metadata_output_file = f"models/{model_type}.metadata.json"
logger.info(f"Writing metadata to: {metadata_output_file}")
with open(metadata_output_file, "w") as metadata_file:
    json.dump(metadata, metadata_file, indent=4)
```

Run training once more, then open `models/random_forest.metadata.json` in PyCharm.

You'll see a classification report broken down by class. Our model predicts two classes:
- `False` — a normal ride
- `True` — an outlier

For each class you'll find:
- **precision** — of all rides the model flagged as outliers, how many actually were?
- **recall** — of all actual outliers in the data, how many did the model catch?
- **f1-score** — the harmonic mean of precision and recall; a single number that balances both

### Step 7: The silent overwrite problem

Run training one more time:

```bash
uv run outlier_detector_training.py random_forest
```

Your `models/random_forest.pkl` and `models/random_forest.metadata.json` were just silently overwritten.

If the two runs produced different results — which one are you keeping? Which parameters did each use?
At what time did each run happen? **You have no idea.** This is the problem that experiment tracking solves.

### 🚀 Level Up

#### Challenge 1: Pickle Jar Museum

Write a small Python script that loads your trained model from disk and runs it on some hand-crafted data:

```python
import pickle
import pandas as pd

with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

rides = pd.DataFrame([
    {"ride_time": 2, "trip_distance": 100},    # 2-second, 100 km trip
    {"ride_time": 1800, "trip_distance": 5},   # 30-minute, 5 km trip
    {"ride_time": 0, "trip_distance": 50},     # Instant 50 km trip
])

predictions = model.predict(rides)
print(predictions)
```

> `"rb"` means "read binary" — the counterpart to `"wb"` when we saved it.

#### Challenge 2: The Chef's Tasting Notes

Regenerate the combined training data using **all days** (not just one):

```bash
uv run combine_taxi_ride_data.py
```

Then train all three model types:

```bash
uv run outlier_detector_training.py random_forest
uv run outlier_detector_training.py random_forest_v2
uv run outlier_detector_training.py logistic_regression
```

Open all three `.metadata.json` files side by side in PyCharm and compare them.

- Which model has the best **recall** for the `True` class (catching real outliers)?
- Which model has the best **precision** for the `True` class (not crying wolf)?
- Which model would you choose for production, and why?


## 🔬 Exercise 4: Experiment Tracking with MLflow

In Exercise 3 you ended up with overwritten files and no history. The fix isn't to rename your
files `random_forest_v1_final_FINAL.pkl`. The fix is a proper experiment tracking system.

[MLflow](https://mlflow.org) is an industry-standard open-source tool for this. It records every
training run — parameters, metrics, the model itself — and gives you a UI to compare them.
Your DagsHub repository already has an MLflow tracking server built in. You just need to connect to it.

### Step 1: Get your DagsHub credentials

You need three pieces of information from your DagsHub repository

`MLFLOW_TRACKING_URI` — the URL of the MLflow tracking server

The format is `https://dagshub.com/<<YOUR_DAGSHUB_USERNAME>>/<<YOUR_DAGSHUB REPOSITORY_NAME>>.mlflow`,

e.g. `https://dagshub.com/peter.rietzler.privat/ais-dev2il-mlops-taxi-rides-v2.mlflow` 

`MLFLOW_TRACKING_USERNAME` — your DagsHub username, e.g. `peter.rietzler.privat`

`MLFLOW_TRACKING_PASSWORD` — your DagsHub access token. This is the same that you already used for DVC

Set them as environment variables in your terminal. On macOS/Linux:

```bash
export MLFLOW_TRACKING_URI=...
export MLFLOW_TRACKING_USERNAME=...
export MLFLOW_TRACKING_PASSWORD=...
```

### Step 2: Add a guard for missing credentials

If any of these variables are missing, the training script will fail with a confusing MLflow error
deep in the stack trace. Add a check at the top of `train_model()` that fails fast with a clear message:

```python
REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]

def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
```

Then call it as the very first thing in `train_model()`:

```python
def train_model(model_type: str):
    check_env_vars()
    ...
```

> 💡 **Fail fast** is a core principle in software engineering: detect problems as early as possible
> and surface a clear error rather than letting them cause mysterious failures later. A missing env var
> is a configuration problem — it should never reach the training code.

### Step 3: Wrap the training in an MLflow run

MLflow organises results into **experiments** (one per model type) and **runs** (one per training execution).

Add the following to `train_model()`, wrapping the existing training dispatch:

```python
mlflow.set_experiment(model_type)
mlflow.autolog()
with mlflow.start_run():
    if model_type == 'random_forest':
        model, metadata = train_random_forest_classifier(DATA_FILE)
    elif model_type == 'random_forest_v2':
        model, metadata = train_random_forest_classifier_v2(DATA_FILE)
    elif model_type == 'logistic_regression':
        model, metadata = train_logistic_regression_classifier(DATA_FILE)
    logger.info("Model training completed")
    logger.info(metadata)
```

Let's break down what's happening:

- **`mlflow.set_experiment(model_type)`** — tells MLflow which experiment this run belongs to.
  An experiment is just a named bucket for related runs. Using `model_type` means each model type
  gets its own experiment in the UI.
- **`mlflow.autolog()`** — one line that automatically records everything scikit-learn knows about
  the training run: hyperparameters, evaluation metrics, the trained model itself, even a feature
  importance plot for tree-based models. This is the magic line.
- **`with mlflow.start_run():`** — opens a new run. Everything that happens inside this block
  gets recorded to MLflow. When the block exits, the run is marked as complete.

### Step 4: Run training and check DagsHub

Run training:

```bash
uv run outlier_detector_training.py random_forest
```

Now open your DagsHub repository and click the **Experiments** tab. Hit **Go to MLflow UI**. 
You should see a `random_forest`
experiment with one run. Click into it — explore the **Parameters**, **Metrics**, and **Logged Models** tabs.

Notice what `autolog()` captured without you writing a single extra line:
- All hyperparameters of the `RandomForestClassifier`
- Accuracy, precision, recall, f1 for both classes
- The trained model itself

Now change the data considered for training using the `combine_taxi_ride_data.py` script to include 3 days, and run the training again. 
Each run gets its own entry. **You can always go back.**

### Step 5: Compare runs

In the MLFlow Experiments tab, select multiple runs and click **Compare**. You'll see a side-by-side
view of all metrics across runs. In order to get more information, select more columns from the "Columns" dropdown 
in the upper right corner. 

🤔 Which model performs best ?

Now run the training for all three model types:

```bash
uv run outlier_detector_training.py random_forest_v2
uv run outlier_detector_training.py logistic_regression
```

In the MLFlow Experiments tab, select multiple experiments and click **Compare**. You'll see a side-by-side
view of all metrics across runs. In order to get more information, select more columns from the "Columns" dropdown 
in the upper right corner. 

🤔 Which model type performs best on your training data? 

### 🚀 Level Up

#### Challenge 1: Attach the Evidence

Right now the `.metadata.json` files live on your local disk — separate from the MLflow run
that produced them. Log it as **artifact** so it's permanently attached to the run.

Add these lines inside `with mlflow.start_run():`, after saving the files:

```python
mlflow.log_artifact(metadata_output_file)
```

Re-run training and find the files under the **Artifacts** tab of your run in the MLFlow UI.

#### Challenge 2: Your Own Metrics

`mlflow.autolog()` logs overall accuracy, but not the per-class breakdown you care about.
Log the class specific metrics manually (also inside the `with mlflow.start_run():` block):

```python
# log some custom metrics
for false_key, false_value in metadata["False"].items():
  mlflow.log_metric(f"False_{false_key}", false_value)
for false_key, false_value in metadata["True"].items():
  mlflow.log_metric(f"True_{false_key}", false_value)
```

Re-run all three model types, look at them and compare the metric values for the 3 runs. 

#### Challenge 3: The MLflow Command Centre 🕹️

> *The browser is great for exploring. The terminal is great for everything else.*

MLflow ships with a CLI that lets you query your tracking server directly from the terminal. The best part: it uses 
**the exact same environment variables** you already have set — `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD`. 
No extra configuration needed.

**List all experiments:**

```bash
uv run mlflow experiments search
```

You should see all three experiments (`random_forest`, `random_forest_v2`, `logistic_regression`) with their IDs and status.

**Get the details of one experiment:**

```bash
uv run mlflow experiments get --experiment-name random_forest
```

This shows information on the experiment. Copy the **Experiment ID** — you'll need it in the next steps.

**List all runs in an experiment:**

```bash
uv run mlflow runs list --experiment-id <EXPERIMENT_ID>
```
You'll see a list of runs. Copy one of the **Run IDs** from the output.

**Inspect a single run in detail:**

```bash
uv run mlflow runs describe --run-id <RUN_ID>
```
This dumps the full JSON of the run — all parameters, all metrics, all tags, and the artifact location. Useful when you need to 
know what a run did without clicking through the UI.

**Export all runs of an experiment to CSV:**

```bash
uv run mlflow experiments csv --experiment-id <EXPERIMENT_ID> --filename runs.csv
```
Open `runs.csv` in PyCharm. You'll have all runs, all metrics, and all parameters in one place — ready for further analysis.

💡 The CLI is especially useful in **automated pipelines**: you can query run IDs, fetch metrics, and make decisions 
(e.g., "only register if recall > 0.9") without opening a browser or writing Python. It is also a great way for any
coding agents to interact with MLFlow.

## 🏆 Exercise 5: MLflow Model Registry — The Hall of Fame

You now have a few runs across three experiments. Some are good. Some are less good. But which one
is actually being used by the application? Without a clear answer to that question, you have a problem.

The **MLflow Model Registry** is the solution. It's a catalogue of promoted, named model versions
with explicit lifecycle stages. Instead of pointing your app at a file path, you point it at a named
model version — and the registry tells you exactly where that version came from.

### Step 1: Register the best model

In the MLFlow Experiments tab, find the `random_forest` run with the best recall for the `True` class.
You can also find the model that was created in the course of this run in the UI (scroll down to the bottom). 
Click on the model and then hit the "Register model" button. Create a new model with the name
`outlier-detection`. 

This promotes the model from "just another run" into a versioned, named entry in the registry.
Every time you register a new version, the version number increments automatically.

### Step 2: Find the version number

Go to the **Models** tab in MLFlow. You'll see your `outlier-detection` model with version `1`.

### Step 3: Create `.model-version`

You need to create a file called `.model-version` in the root of your repository. If that sounds
familiar — it's exactly the same idea as `.python-version`, which `uv` uses to know which Python
version to use. This file pins which model version your application should use.

Create the file and put just the version number inside:

```
1
```

> 💡 Plain text files that contain a single version number are a lightweight but powerful pattern.
> They make a deployment decision **explicit, traceable, and reviewable** — changing the model
> version becomes a normal Git commit, visible in history and reviewable in a pull request.

This tiny file is the contract between the registry and the application. It says:
*"When you deploy this commit, then version 1 of the `outlier-detection` model is used"*

### Step 4: Download the model from the registry

There is already a file called `download_model.py` in the repository. Open it in PyCharm and
read through it. The key lines are these:

```python
version = open(".model-version").read().strip()
model_uri = f"models:/outlier-detection/{version}"

model = mlflow.sklearn.load_model(model_uri)

with open("outlier_detection_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

Let's unpack the MLFlow model URI: `models:/outlier-detection/1`

- **`models:/`** — tells MLflow to look in the **Model Registry** (not in a run's artifacts)
- **`outlier-detection`** — the registered model name you chose in Step 1
- **`1`** — the version number, read from `.model-version`

The script uses the same `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
environment variables to know where to find the registry — the same ones you've been using all along.

Run it:

```bash
uv run download_model.py
```

You'll find `outlier_detection_model.pkl` in the root of your repository — the same pickle format
you created in Exercise 3, but now fetched from the registry rather than from a local training run.

### Step 5: Serve predictions with FastAPI

You have a model. Now let's put it to work. Open `outlier_detection_api.py` in PyCharm and read through it. The two parts that matter:

**Loading the model at startup:**

```python
with open("outlier_detection_model.pkl", "rb") as f:
    model = pickle.load(f)
```

This runs once when the server starts. It reads the pickle file you just downloaded and keeps the model in memory — ready to make predictions 
without reloading from disk on every request.

**Making a prediction:**

```python
@app.get("/detect-outliers", response_model=OutlierDetectionResponse)
def detect_outliers(
    ride_time: float = Query(...),
    trip_distance: float = Query(...)
):
    input_df = pd.DataFrame([{"ride_time": ride_time, "trip_distance": trip_distance}])
    preds = model.predict(input_df)
    return OutlierDetectionResponse(outlier=bool(preds))
```

This is an HTTP endpoint that accepts `ride_time` and `trip_distance` as query parameters, 
runs them through the model, and returns a JSON response with a single field: `outlier` — `true` or `false`.

**Start the server:**

```bash
uv run fastapi dev outlier_detection_api.py
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Try it out in the browser:**

FastAPI automatically generates an interactive API documentation page. Open:

```
http://127.0.0.1:8000/docs
```

You'll see the `/detect-outliers` endpoint. Click on it → **Try it out** → fill in some values and hit **Execute**:

- `ride_time: 2`, `trip_distance: 100` — a 2-second, 100-mile trip. Outlier? 🤔
- `ride_time: 1800`, `trip_distance: 5` — a 30-minute, 5-mile trip. Normal? 🤔
- `ride_time: 0`, `trip_distance: 50` — an instant 50-mile trip. Definitely an outlier! 🚨

The page shows you the exact URL it called and the raw JSON response — great for understanding how the API works under the hood.

Press `Ctrl+C` in the terminal to stop the server when you're done.

### Step 6: The full lifecycle

Here's the complete workflow you just ran through, end to end:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MLOps Workflow Summary                          │
└─────────────────────────────────────────────────────────────────────┘

  1. 📦 Prepare data
     uv run combine_taxi_ride_data.py 2025-01-01
     → data/taxi-rides-training-data.parquet

  2. 🏋️ Train the model
     uv run outlier_detector_training.py random_forest
     → run logged to MLflow (params, metrics, model artifact)

  3. 🔬 Select the best run
     Open MLflow UI on DagsHub → Experiments
     → compare runs, pick the one with the best recall

  4. 🏆 Register the model
     MLflow UI → Artifacts → Register Model → "outlier-detection"
     → outlier-detection vX in the Model Registry

  5. 📌 Pin the version
     echo "X" > .model-version
     git commit -m "Deploy outlier-detection v1"
     → version decision is explicit and tracked in Git

  6. ⬇️  Download the model
     uv run download_model.py
     → outlier_detection_model.pkl

  7. 🚀 Serve predictions
     uv run fastapi dev outlier_detection_api.py
     → http://127.0.0.1:8000/docs
```

Every step is reproducible: the data is versioned in DVC, the training run is logged in MLflow,
the model version is pinned in Git, and the API always loads exactly the model you chose.


### 🚀 Level Up

#### Challenge 1: Unbox the Model 📦

In Exercise 4 you used the MLflow CLI to inspect experiments and runs. 
Now use it to **download the registered model** directly from the Model Registry — no Python script needed.

The `--artifact-uri` flag accepts the same `models:/` URI format you already know from `download_model.py`:

```bash
uv run mlflow artifacts download --artifact-uri models:/outlier-detection/1 --dst-path downloaded_model
```

Take a look at what's inside:

```bash
ls downloaded_model/
```

You'll find `model.pkl`, `MLmodel` (the model metadata), and `requirements.txt` (the exact Python dependencies needed to load it). This is what MLflow actually stores in the registry — more than just the pickle file.

Open `MLmodel` in PyCharm. It describes the model: the flavours it supports (e.g. `sklearn`, `python_function`), 
the Python version it was trained with, and the path to the pickle file inside the directory.

#### Challenge 2: Promote a New Champion 🥊

So far you've trained on a single day of data. Let's train a better model, register it as version 2, and deploy it.

Before you start, make sure your working directory is clean and all previous changes are pushed.

**Regenerate the training data with more days:**

```bash
uv run combine_taxi_ride_data.py 2025-01-01 2025-01-02 2025-01-03
uv run dvc add data/taxi-rides-training-data.parquet
uv run dvc push
git add data/taxi-rides-training-data.parquet.dvc
git commit -m "Update training data: 3 days"
git push
```

**Retrain:**

```bash
uv run outlier_detector_training.py random_forest
```

**Register the new run** as a new version of `outlier-detection` in the MLflow UI (same as Step 1 — the version number will automatically become `2`).

**Bump `.model-version`:**

```
2 (OR WHATEVER THE NEW VERSION NUMBER IS)
```

**Re-download and restart:**

```bash
uv run download_model.py
uv run fastapi dev outlier_detection_api.py
```

Try some predictions as before in the Swagger UI at `http://127.0.0.1:8000/docs`.

Finally also push the model bump: 

```bash
git add .model-version
git commit -m "Deploy outlier-detection v2"
git push
```

#### Challenge 3: Long Live the Champion 👑

Pinning a version number in `.model-version` works — but what does `2` mean six months from now? MLflow Registry supports **aliases**: 
named pointers to a specific version. You can e.g. use `@champion` for the model currently in production.

**Set the alias in the MLflow UI:**

Go to the **Models** tab → `outlier-detection` → click on the version you want to promote → find the **Aliases** field and add `champion`.

**Update `download_model.py` to use the alias instead of the version number:**

Find this line:

```python
model_uri = f"models:/{MODEL_NAME}/{version}"
```

And change it to:

```python
model_uri = f"models:/{MODEL_NAME}@champion"
```

You can now delete `.model-version` entirely and remove the line that reads it.

**Run it:**

```bash
uv run download_model.py
```

It still works — but now it always pulls whatever version is tagged `@champion`. Promoting a new model to production is just moving the alias 
in the UI — no code change, no file update, no commit needed.

💡 Notice the trade-off: with `.model-version`, every deployment decision was a Git commit — visible in history, reviewable in a pull request. 
With aliases, that traceability moves out of Git and into MLflow. Promoting a new champion leaves no trace in your repository. Whether that's 
acceptable depends on your team's process — but it's worth being aware of.
---

## 🤖 Exercise 6: The Training Pipeline

In the previous exercises you trained the model by hand — running `combine_taxi_ride_data.py` and `outlier_detector_training.py` 
locally on your machine. That works, but in Dev/MLOPs we aim for automation - right ? Let's see how we can automate the training
using techniques that we already know and love.

### Step 1: Create the workflow file

Create the file `.github/workflows/train.yml`.

### Step 2: Define the trigger

We want the pipeline to run whenever code is pushed to `main`, and we also want to be able to trigger it manually.

```yaml
name: Train Models

on:
  push:
    branches: [main]
  workflow_dispatch:
```

You've seen both of these before in the CI exercise - so no need to dig deep here.

### Step 3: Train all three models in parallel

Here's the interesting part. Instead of writing three separate jobs (one for `random_forest`, one for `random_forest_v2`, one for `logistic_regression`), 
GitHub Actions lets you define **one job** and run it multiple times with different values — a **matrix strategy**.

```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        model_type: [random_forest, random_forest_v2, logistic_regression]
```

Think of it like a loop: GitHub reads the list and spins up one job per value, all running in parallel. Inside the job you can 
refer to the current value with `${{ matrix.model_type }}`.

`fail-fast: false` means that if one model fails to train, the other two keep going. Without it, a single failure would cancel 
all running jobs immediately.

### Step 4: Configure credentials

The training script needs to connect to MLflow on DagsHub. 
We therefore have to set the three environment variables you've been exporting locally.

Add them to the job:

```yaml
    env:
      MLFLOW_TRACKING_URI: ${{ vars.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ vars.MLFLOW_TRACKING_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
```

Notice the difference between `vars.*` and `secrets.*`:
- **`vars`** — plain text values, visible in the UI. Use for non-sensitive configuration like URLs and usernames.
- **`secrets`** — encrypted, never shown in logs or the UI. Use for passwords and tokens.

You need to add these to your repository settings. You already know how to do this from the CI exercise — go to **Settings → Secrets and variables → Actions**
and configure everything you need.

<details>
<summary>🆘 Lost? Click here for step-by-step instructions</summary>

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Under **Variables**, click **New repository variable** and add:
   - `MLFLOW_TRACKING_URI` — e.g. `https://dagshub.com/<your-username>/<your-repo>.mlflow`
   - `MLFLOW_TRACKING_USERNAME` — your DagsHub username
4. Under **Secrets**, click **New repository secret** and add:
   - `MLFLOW_TRACKING_PASSWORD` — your DagsHub access token (the same one you used for DVC)

</details>

### Step 5: Add the job steps

Now add the steps that actually do the work:

```yaml
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync

      - name: Pull data with DVC
        run: |
          uv run dvc remote modify --local origin access_key_id ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
          uv run dvc remote modify --local origin secret_access_key ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
          uv run dvc pull

      - name: Train model (${{ matrix.model_type }})
        run: uv run outlier_detector_training.py ${{ matrix.model_type }}
```

The DVC step injects your DagsHub token as the S3 credentials using `--local` (same as you did manually in Exercise 2 — 
but here it reads from the secret instead of your terminal). 
`${{ matrix.model_type }}` passes the current matrix value as the argument to the training script.

### Step 6: Trigger the pipeline

Commit and push your new workflow file.

Watch three jobs spin up in parallel — one for each model type.  
When they complete, check your DagsHub MLflow UI — you should see new runs logged for all three experiments.

🎉 You just trained three models in parallel in the cloud.

### 🚀 Level Up

#### Challenge 1: Only Train When It Matters 🎯

Right now the pipeline runs on every push to `main` — even if you only changed a comment in `README.md`. That wastes CI minutes and 
clutters your MLflow with unnecessary runs.

GitHub Actions supports **path filters**: the workflow only runs if at least one of the listed paths was changed in the push.

Update the `push` trigger:

```yaml
on:
  push:
    branches: [main]
    paths:
      - "data/**"
      - "uv.lock"
      - "outlier_detector_training.py"
  workflow_dispatch:
```

- `data/**` — any change to a DVC pointer file (new or updated data)
- `uv.lock` — dependency changes that could affect training
- `outlier_detector_training.py` — the training script itself changed

Commit and push this change. Now try pushing a small README edit — the workflow should **not** trigger. Then push a change to `outlier_detector_training.py` — it should trigger.

💡 `workflow_dispatch` always works regardless of path filters. So you can still trigger manually whenever you need to.

---

## 🐳 Exercise 7: Package and Ship the API

You can run the API on your machine with `uv run fastapi dev`. But how do you ship it somewhere else — a server, a teammate's machine, a cloud environment? 
You can't just send your laptop. Docker is the answer. You already know how to build images. Let's package this API.

### Block 1: The Dockerfile

Create a file called `Dockerfile` in the root of your repository and paste in the following:

```dockerfile
ARG PYTHON_VERSION=3.13

# Builder stage: Install dependencies
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# Install curl for uv
RUN apt-get update && apt-get install -y curl

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files needed for dependency installation
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies (creates .venv with Python and packages)
RUN uv sync --frozen

# Runtime stage: Clean image with only runtime dependencies
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Copy Python environment and application from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY outlier_detection_api.py ./
COPY outlier_detection_model.pkl ./

# Set the entrypoint
ENTRYPOINT ["/app/.venv/bin/fastapi", "run", "outlier_detection_api.py"]
```

This uses the same multi-stage builder pattern you already know from the Docker exercises — nothing new there.

The one line to pay attention to:

```dockerfile
COPY outlier_detection_model.pkl ./
```

The trained model file gets **baked into the image**. The resulting image is completely self-contained — it needs no 
MLflow connection, no download script, no environment variables at runtime. Everything needed to serve predictions is inside the image.

This means you must have `outlier_detection_model.pkl` on disk **before** you run `docker build`. Run `download_model.py` first if you haven't already.

**Build the image:**

```bash
docker build -t outlier-detection-api .
```

**Run it locally:**

```bash
docker run --rm -p 8000:8000 outlier-detection-api
```

Open `http://127.0.0.1:8000/docs` and try a prediction. Same API, now running inside a container.

### Block 2: The CI Pipeline

Doing this manually every time `.model-version` changes gets old fast. Let's automate it.

Create `.github/workflows/api.yml` and paste in the following:

```yaml
name: Build Outlier Detection API Server

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-push:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    permissions:
      packages: write

    env:
      MLFLOW_TRACKING_URI: ${{ vars.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ vars.MLFLOW_TRACKING_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Install dependencies
        run: uv sync --frozen

      - name: Download model from MLflow registry
        run: uv run download_model.py

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Read Python version
        id: python-version
        run: echo "version=$(cat .python-version)" >> $GITHUB_OUTPUT

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:v${{ github.run_number }}
          build-args: |
            PYTHON_VERSION=${{ steps.python-version.outputs.version }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

The key thing to notice is the **step order**:

1. `download_model.py` runs first — this fetches `outlier_detection_model.pkl` from the MLflow registry onto the CI runner
2. `docker build` runs after — this bakes the `.pkl` file into the image

It's exactly the same sequence you just did manually. CI just does it automatically on every push.

The image gets two tags:
- `latest` — always points to the most recent build
- `v<run_number>` — a pinned, traceable version (e.g. `v42`) you can always roll back to

**Trigger the pipeline:**

Commit and push your `api.yml`. Watch the workflow run. When it completes, go to your repository on 
GitHub check if the package is there.

### 🚀 Level Up

#### Challenge 1: Run the Ship 🚢

Pull the image that CI just built and run it on your local machine:

```bash
docker pull ghcr.io/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>:latest
docker run --rm -p 8000:8000 ghcr.io/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>:latest
```

❓Having "Platform" issues ("no matching manifest ..." errors). Remember this [section](https://github.com/peterrietzler/ais-dev2il-ais-power-smoothie-delivery-box#-troubleshooting-platform-compatibility-issues) 
from our last sessions.

Open `http://127.0.0.1:8000/docs`. You're now running the exact image that CI built — not a local build, not a dev server. The real thing.

#### Challenge 2: Only Ship When It Matters 🎯

Right now the pipeline rebuilds the Docker image on every push to `main`. That's wasteful — a README change shouldn't trigger a Docker build.

Add a `paths:` filter so the pipeline only runs when something actually affecting the image changes:

```yaml
on:
  push:
    branches: [main]
    paths:
      - ".model-version"
      - "Dockerfile"
      - "outlier_detection_api.py"
      - "pyproject.toml"
      - "uv.lock"
  workflow_dispatch:
```

Push a README change and confirm the pipeline does **not** trigger. Then bump `.model-version` and confirm it **does**.

#### Challenge 3: Image Detective 🔍

Confirm the model file is actually inside the image:

```bash
docker run -it --entrypoint sh --rm ghcr.io/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>:latest
```

When you list the files in the current working directory, you should see `outlier_detection_model.pkl` 
listed alongside `outlier_detection_api.py`. The model is in the box. 