import pandas as pd  # Data manipulation (DataFrames)
import numpy as np   # Numerical operations (Arrays, Log math)
import matplotlib.pyplot as plt # Visualization library
import random # For generating random negative samples
import ast # "Abstract Syntax Tree" - safely parses string lists "['a','b']" into actual Python lists
from collections import defaultdict # Dictionaries that handle missing keys automatically
from sklearn.preprocessing import StandardScaler # For Z-score normalization (Scaling)
from sklearn.linear_model import LogisticRegression # The classification model
import math, random #self-explanatory
from sklearn.metrics import roc_auc_score, accuracy_score #Used for evaluation
# Magic command to display plots directly in the notebook
#%matplotlib inline 

# ==========================================
# 0. PATH SETUP
# ==========================================
# Define the folder path containing all dataset files.
# NOTE: This path matches my local computer's folder structure, might need to readjust for yours
base_path = 'C:/Users/avina/Desktop/School Work/CSE 158/assignment2/'

print(f"Working directory set to: {base_path}")

# ==========================================
# PART 1: EXPLORATORY DATA ANALYSIS (EDA)
# Visualize data distribution to justify our feature engineering choices (e.g., Log-Transform).
# ==========================================
print("\n--- PART 1: EXPLORATORY ANALYSIS ---")

try:
    # Load RAW CSVs. Use RAW files here because they have readable columns (Ratings, Dates).
    raw_inter_df = pd.read_csv(base_path + 'RAW_interactions.csv')
    raw_recipes_df = pd.read_csv(base_path + 'RAW_recipes.csv')
    
    # --- 1.1 Sparsity Calculation ---
    # Sparsity defines how "hard" the recommendation task is.
    # High Sparsity (>99%) means users have interacted with almost 0% of available items.
    num_users = raw_inter_df['user_id'].nunique()
    num_items = raw_inter_df['recipe_id'].nunique()
    sparsity = len(raw_inter_df) / (num_users * num_items)
    print(f"Sparsity: {sparsity:.6f}")
    #Interpretation: The matrix is extremely sparse. This implies collaborative filtering would struggle so we used features
    # --- 1.2 Visualizations ---
    plt.figure(figsize=(15, 4))
    
    # Plot 1: Rating Distribution
    # WHY: To determine if we should predict Stars (1-5) or Interactions (0/1).
    plt.subplot(1, 4, 1)
    raw_inter_df['rating'].hist(bins=10, color='skyblue', edgecolor='black')
    plt.title("Ratings Distribution")
    # CONCLUSION: Most ratings are 5 stars. Predicting "Did they click?" is more valuable than "How many stars?".

    # Pre-processing for Plots: Fill NaNs with 0 to avoid errors
    raw_recipes_df['n_ingredients'] = raw_recipes_df['n_ingredients'].fillna(0)
    raw_recipes_df['minutes'] = raw_recipes_df['minutes'].fillna(0)
    # Extract Calories from the string "[500.0, ...]"
    raw_recipes_df['calories'] = raw_recipes_df['nutrition'].apply(
        lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else 0
    )

    # Plot 2: Ingredients
    plt.subplot(1, 4, 2)
    raw_recipes_df['n_ingredients'].hist(bins=30, color='orange', edgecolor='black')
    plt.title("# Ingredients")
    
    # Plot 3: Cooking Time (Log Scale)
    # Log because some recipes have 10,000+ minutes (errors/outliers). 
    # Log-transforming squashes outliers so we can see the true distribution.
    plt.subplot(1, 4, 3)
    np.log1p(raw_recipes_df['minutes']).hist(bins=30, color='green', edgecolor='black')
    plt.title("Time (Log Scale)")
    
    # Plot 4: Calories (Log Scale)
    plt.subplot(1, 4, 4)
    np.log1p(raw_recipes_df['calories']).hist(bins=30, color='red', edgecolor='black')
    plt.title("Calories (Log Scale)")
    
    plt.tight_layout()
    plt.show() # Renders the plot window 


    

except FileNotFoundError:
    print("Warning: RAW files not found. Skipping EDA.")



# ==========================================
# PART 2: DATA LOADING & FEATURE ENGINEERING
# GOAL: Prepare data for the model. Includes "Bridging" IDs and Normalizing features.
# ==========================================
print("\n--- PART 2: MODELING PIPELINE ---")
print("Loading Official Train/Test Splits...")

# Load the specific files required
# 'interactions_train.csv' uses mapped integers (0, 1, 2...) for User/Item IDs.
train_df = pd.read_csv(base_path + 'interactions_train.csv')
test_df = pd.read_csv(base_path + 'interactions_test.csv')
pp_recipes = pd.read_csv(base_path + 'PP_recipes.csv') # Pre-Processed Mapping File

# Reload RAW recipes for feature extraction (Tags, Nutrition)
if 'raw_recipes_df' not in locals():
    raw_recipes_df = pd.read_csv(base_path + 'RAW_recipes.csv')
    raw_recipes_df['n_ingredients'] = raw_recipes_df['n_ingredients'].fillna(0)
    raw_recipes_df['minutes'] = raw_recipes_df['minutes'].fillna(0)
    raw_recipes_df['calories'] = raw_recipes_df['nutrition'].apply(
        lambda x: float(ast.literal_eval(x)[0]) if isinstance(x, str) else 0
    )

# --- 2.2 ID Mappings (The "Bridge") ---
# PROBLEM: The Model sees processed IDs ('i'), but Features are in Raw IDs ('id').
# SOLUTION: Create dictionaries to map between them.

# Map: Processed ID ('i') -> Raw ID ('id')
i_to_raw_id = pd.Series(pp_recipes.id.values, index=pp_recipes.i).to_dict()

# Map: Processed ID -> Set of Ingredient IDs (For Jaccard Similarity)
# We convert string lists "['1','2']" to Python Sets {1, 2} for fast intersection math.
pp_recipes['ingr_ids_list'] = pp_recipes['ingredient_ids'].apply(ast.literal_eval)
i_to_ingr_set = pd.Series(pp_recipes['ingr_ids_list'].values, index=pp_recipes.i).to_dict()

# Map: Raw ID -> Set of Tags (For Jaccard Similarity)
# We convert "['mexican', 'easy']" to Python Sets {'mexican', 'easy'}.
raw_recipes_df['tags_list'] = raw_recipes_df['tags'].apply(lambda x: set(ast.literal_eval(x)))
raw_id_to_tags = pd.Series(raw_recipes_df['tags_list'].values, index=raw_recipes_df['id']).to_dict()

# --- 2.3 Feature Normalization ---
# WHY: "Minutes" ranges 0-5000, "Ingredients" ranges 0-20. 
# Without normalization, the model will assume "Minutes" is 100x more important.
print("Normalizing Content Features...")

# Step A: Log Transform (np.log1p) to handle skewed data/outliers
raw_recipes_df['log_minutes'] = np.log1p(raw_recipes_df['minutes'])
raw_recipes_df['log_calories'] = np.log1p(raw_recipes_df['calories'])
raw_recipes_df['log_n_ingr'] = np.log1p(raw_recipes_df['n_ingredients'])

# Step B: StandardScaler (Z-Score)
# Rescales data so Mean = 0 and Std Dev = 1.
scaler = StandardScaler()
cols = ['log_minutes', 'log_calories', 'log_n_ingr']
norm_feats = scaler.fit_transform(raw_recipes_df[cols])

# Create Map: Raw ID -> [Normalized Features]
raw_id_to_feats = {rid: norm_feats[idx] for idx, rid in enumerate(raw_recipes_df['id'])}

# --- 2.4 Master Feature Dictionary ---
# Consolidate all features into dictionaries keyed by the Training ID ('i')
item_content_feats = {} # Stores [Min, Cal, Ingr]
item_tag_sets = {}      # Stores {Tags}
valid_items = []        # List of items that have valid features

for i, raw_id in i_to_raw_id.items():
    if raw_id in raw_id_to_feats:
        item_content_feats[i] = raw_id_to_feats[raw_id]
        valid_items.append(i)
    if raw_id in raw_id_to_tags:
        item_tag_sets[i] = raw_id_to_tags[raw_id]

# 2.5 Popularity Dictionary (For Baseline Comparison ONLY)
# Counts how often each item appears in Train
item_counts = train_df['i'].value_counts().to_dict()


# ==========================================
# VISUALIZATION: TOP 20 MOST POPULAR (BASELINE)
# ==========================================
# Refractored popularity baseline

def show_most_popular_items(top_n=20):
    print(f"--- TOP {top_n} MOST POPULAR RECIPES (BASELINE) ---")
    # 2. Mapping using code
    # Map: Processed ID (i) -> Raw ID (id)
    print("Building ID maps...")

    df_raw = raw_recipes_df
    local_i_to_raw = pd.Series(pp_recipes['id'].values, index=pp_recipes['i']).to_dict()
    
    # Map: Raw ID (id) -> Recipe Name
    local_raw_to_name = pd.Series(df_raw['name'].values, index=df_raw['id']).to_dict()

    # 3. Get Top Counts from Train
    top_items = train_df['i'].value_counts().head(top_n)
    
    print(f"\n{'Rank':<5} | {'Count':<8} | {'Recipe Name'}")
    print("-" * 65)
    
    for rank, (i, count) in enumerate(top_items.items()):
        if i in local_i_to_raw:
            raw_id = local_i_to_raw[i]
            name = local_raw_to_name.get(raw_id, "Unknown Name")
            
            # Truncate long names
            if isinstance(name, str) and len(name) > 45: 
                name = name[:42] + "..."
            
            print(f"#{rank+1:<4} | {count:<8} | {name}")
        else:
            print(f"#{rank+1:<4} | {count:<8} | [ID {i} not found]")

# Run it
show_most_popular_items(20)


# ==========================================
# VISUALIZATION 2: THE "GHOST" RECIPES (ZERO HISTORY)
# ==========================================
# Shows why the popularity baseline fails (0% hit rate).

def show_popularity_dropoff(top_n=20):
    print(f"\n--- POPULARITY DROPOFF ANALYSIS (Top {top_n} Train Items) ---")
    
    # 1. Get Top Items from Training Set
    # item_counts is the dictionary {item_id: count} from Part 2
    top_train_items = pd.Series(item_counts).sort_values(ascending=False).head(top_n)
    
    # 2. Calculate how many times these appear in TEST Set
    # We filter the test dataframe for these specific IDs
    test_counts_series = test_df['i'].value_counts()
    
    # 3. Setup Name Lookup
    raw_id_to_name = pd.Series(raw_recipes_df['name'].values, index=raw_recipes_df['id']).to_dict()
    
    print(f"{'Rank':<5} | {'Recipe Name (Train Leader)':<40} | {'Train Reviews':<12} | {'Test Reviews'}")
    print("-" * 85)
    
    for rank, (i, train_count) in enumerate(top_items.items()):
        # Get Test Count (Default to 0 if nobody clicked it in Test)
        test_count = test_counts_series.get(i, 0)
        
        # Get Name
        if i in i_to_raw_id:
            raw_id = i_to_raw_id[i]
            name = raw_id_to_name.get(raw_id, "Unknown Name")
            if len(name) > 30: name = name[:25] + "..."
        else:
            name = f"ID {i}"
            
        # Highlight rows where Test Count is ZERO
        alert = " <--- zero times!" if test_count == 0 else ""
        
        print(f"#{rank+1:<4} | {name:<40} | {train_count:<12} | {test_count}{alert}")

# Run it
# Note: We re-calculate top_items inside the function to be safe
top_items = train_df['i'].value_counts().head(20)
show_popularity_dropoff(20)

# ==========================================
# PART 3: MODEL TRAINING
# GOAL: Train Logistic Regression to predict clicks based on Feature Compatibility.
# ==========================================
print("\n--- PART 3: TRAINING MODEL ---")

# --- 3.1 Build User Profiles ---
# STRATEGY: "Centroid" Profiling.
# We represent a User as the AVERAGE of all items they interacted with in the Training set.
print("Building User Profiles...")
user_centroids = {} 
user_ingr_sets = defaultdict(set) # Union of all ingredients user liked
user_tag_sets = defaultdict(set)  # Union of all tags user liked
user_sums = defaultdict(lambda: np.zeros(3))
user_cnts = defaultdict(int)

for _, row in train_df.iterrows():
    u = row['u']
    i = row['i']
    
    # Accumulate Numerical Features
    if i in item_content_feats:
        user_sums[u] += item_content_feats[i]
        user_cnts[u] += 1
    # Accumulate Ingredients
    if i in i_to_ingr_set:
        user_ingr_sets[u].update(i_to_ingr_set[i])
    # Accumulate Tags
    if i in item_tag_sets:
        user_tag_sets[u].update(item_tag_sets[i])

# Compute Averages
for u in user_sums:
    if user_cnts[u] > 0:
        user_centroids[u] = user_sums[u] / user_cnts[u]

# --- 3.2 Generate Training Data (Negative Sampling) ---
# WHY: We have "One-Class" data (Only Reviewed). We need "Non-Reviewed" to train a classifier.
# STRATEGY: For every True Click (1), we pick a Random Recipe (0).
print("Generating Training Samples...")
X_train = []
y_train = []
train_sample = train_df.sample(n=50000, random_state=42) # Sample 50k for speed

def get_content_vector(u, i):
    """
    Constructs the 5-dimensional Feature Vector for (User, Item).
    1. Ingredient Jaccard (Taste Match)
    2. Tag Jaccard (Theme Match)
    3. Time Diff (Lifestyle Match)
    4. Calorie Diff (Diet Match)
    5. Complexity Diff (Skill Match)
    """
    # 1. Ingredient Jaccard
    jaccard_ingr = 0.0
    if u in user_ingr_sets and i in i_to_ingr_set:
        u_s = user_ingr_sets[u]
        i_s = i_to_ingr_set[i]
        if u_s and i_s:
            jaccard_ingr = len(u_s.intersection(i_s)) / len(u_s.union(i_s))

    # 2. Tag Jaccard
    jaccard_tags = 0.0
    if u in user_tag_sets and i in item_tag_sets:
        u_t = user_tag_sets[u]
        i_t = item_tag_sets[i]
        if u_t and i_t:
            jaccard_tags = len(u_t.intersection(i_t)) / len(u_t.union(i_t))

    # 3, 4, 5. Absolute Differences
    if u in user_centroids and i in item_content_feats:
        diff = np.abs(user_centroids[u] - item_content_feats[i])
    else:
        diff = np.array([0.0, 0.0, 0.0]) # Fallback for cold users
        
    return np.concatenate(([jaccard_ingr, jaccard_tags], diff))

for _, row in train_sample.iterrows():
    u = row['u']
    pos_i = row['i']
    
    if u not in user_centroids: continue
    
    # Positive Sample
    X_train.append(get_content_vector(u, pos_i))
    y_train.append(1)
    
    # Negative Sample
    neg_i = random.choice(valid_items)
    while neg_i == pos_i: 
        neg_i = random.choice(valid_items)
        
    X_train.append(get_content_vector(u, neg_i))
    y_train.append(0)

# --- 3.3 Train Model ---
print("Fitting Logistic Regression...")
# Config: 
# class_weight='balanced': Treats 0s and 1s as equally important.
# C=1.0: Regularization strength (Standard default).
model = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0)
model.fit(X_train, y_train)

# Print Weights for Video
weights = model.coef_[0]
print("--- Model Weights Interpretation ---")
print(f"Ingr Jaccard:   {weights[0]:.4f} (Positive = Users like shared ingredients)")
print(f"Tag Jaccard:    {weights[1]:.4f}) (Should be Positive, but the Users apparently appreciate variety??")
print(f"Time Diff:      {weights[2]:.4f} (Negative = Users dislike Time mismatch)")
print(f"Calorie Diff:   {weights[3]:.4f} (Negative = Users dislike Calorie mismatch)")
print(f"Complexity Diff:{weights[4]:.4f} (Negative = Users dislike Complexity mismatch)")

# ==========================================
# PART 4: EVALUATION
# GOAL: Compare Model Accuracy vs Popularity Baseline on Test Set.
# ==========================================
print("\n--- PART 4: EVALUATION ---")

# Define Global Top 10 (Static Baseline)
sorted_items = sorted(item_counts, key=item_counts.get, reverse=True)
global_top_10 = set(sorted_items[:10])

# ==========================================
# 5. FINAL EVALUATION (ALL METRICS)
# ==========================================
import math
import random
from sklearn.metrics import roc_auc_score

def evaluate_all_metrics(test_data, limit=1000):
    """
    Evaluates the model using the standard '1 vs 99' Ranking Protocol.
    Calculates ALL key metrics at once.
    
    METRICS EXPLAINED:
    1. HR@10 (Hit Rate): Did the True Item appear on the 'First Page' (Top 10)?
    2. NDCG@10: Did the True Item appear HIGH on the list? (Rank 1 > Rank 10).
    3. AUC: Probability that the Model scores the True Item higher than a Random Decoy.
    """
    
    # --- 1. Setup Accumulators (To calculate averages later) ---
    # We will add up the scores for every single user here.
    total_users = 0
    
    # Accumulators for Our Content Model
    sum_hr_model = 0.0   # Sum of Hit Rates
    sum_ndcg_model = 0.0 # Sum of NDCG scores
    sum_auc_model = 0.0  # Sum of AUC scores
    
    # Accumulators for the Popularity Baseline (Comparison)
    sum_hr_pop = 0.0
    sum_ndcg_pop = 0.0
    sum_auc_pop = 0.0
    
    print(f"Evaluating on random subset of {limit} users...")
    print("Protocol: For each user, rank the True Item against 99 Random Decoys.")
    
    # Downsample the test set for speed (Efficiency)
    # random_state=42 ensures we get the same 'random' users every time we run this.
    subset = test_data.sample(n=limit, random_state=42)
    
    # --- 2. The Evaluation Loop ---
    # Iterate through each test user one by one.
    for _, row in subset.iterrows():
        u = row['u']          # The User ID
        true_item = row['i']  # The Recipe ID they actually clicked (The "Ground Truth")
        
        # Is this a "Cold Start" User? Cold Start user is someone with no history in training btw
        # If the user has NO history in the training set, we cannot compute features 
        # like "Ingredient Compatibility" or "Time Difference".
        # We must detect this so we can fallback to a simple baseline for them.
        is_cold_start = u not in user_centroids
        
        # ============================================================
        # STEP A: Candidate Generation (Creating the "Search Results")
        # ============================================================
        # To simulate a real ranking task, we need a list of items to sort.
        # We use the "1 vs 99" protocol (Standard in Research).
        
        # 1. Start the list with the ONE item the user actually liked.
        candidates = [true_item]
        
        # 2. Create a list of "Labels" (Truth values) for metrics later.
        # 1 = Real Item (The Target)
        # 0 = Decoy Item (Noise)
        labels = [1] 
        
        # 3. Fill the rest of the list with 99 Random Decoys. https://arxiv.org/abs/1708.05031 for the source here. I'm not too certain on why we do this myself.
        while len(candidates) < 100:
            # Pick a random item from the entire catalog
            decoy = random.choice(valid_items)
            
            # CRITICAL CHECK: Ensure we don't accidentally pick the True Item again.
            if decoy != true_item:
                candidates.append(decoy) # Add decoy to the list
                labels.append(0)         # Mark it as a "0" (Fake)
        
        # ============================================================
        # STEP B: Scoring (Model vs Baseline)
        # ============================================================
        # Now we have 100 items. We need to ask the model: "How good are these?"
        
        model_scores = [] # To store Our Model's probabilities
        pop_scores = []   # To store Popularity Baseline's counts
        
        # Loop through all 100 candidates (1 Real + 99 Decoys)
        for item in candidates:
            
            # --- 1. Calculate Content Model Score ---
            if not is_cold_start:
                # CASE: WARM USER (We know them)
                # We calculate the 5 features (Jaccard, Time Diff, Calorie Diff, etc.)
                # .reshape(1, -1) is needed because sklearn expects a batch, not a single row.
                vec = get_content_vector(u, item).reshape(1, -1)
                
                # Ask Logistic Regression for a probability (0.0 to 1.0)
                # predict_proba returns [[Prob_Class_0, Prob_Class_1]]
                # We take [0][1] because we want the probability of Class 1 (Click).
                score = model.predict_proba(vec)[0][1]
            else:
                # CASE: COLD USER (Stranger)
                # We have no data, so we cannot use the fancy model.
                # Fallback: Just use the item's popularity count.
                score = item_counts.get(item, 0)
                
            model_scores.append(score)
            
            # --- 2. Calculate Popularity Baseline Score ---
            # For the baseline, the "score" is just the raw popularity count.
            # .get(item, 0) returns 0 if the item was never seen in training.
            pop_scores.append(item_counts.get(item, 0))
            
        # ============================================================
        # STEP C: Calculate Metrics for this User
        # ============================================================
        # Now we see where the True Item landed in the rankings.
        
        # We define a helper function to calculate HR, NDCG, and AUC for any list of scores.
        def calc_user_metrics(scores, current_labels):
            # 1. Pair the Scores with the Labels so we can sort them together.
            # Structure: [(Score, Label), (Score, Label)...]
            zipped = list(zip(scores, current_labels))
            
            # 2. Sort the list by Score in Descending Order (Highest Confidence first).
            zipped.sort(key=lambda x: x[0], reverse=True)
            
            # 3. Find the Rank of the True Item (Label = 1).
            rank = -1
            for r, (score, label) in enumerate(zipped):
                if label == 1:
                    # We found it!
                    # r is 0-indexed (0, 1, 2...), so we add 1 to make it "Human" (1st, 2nd...)
                    rank = r + 1 
                    break
            
            # 4. Calculate Hit Rate @ 10
            # Did it appear in the top 10 positions?
            if rank <= 10:
                hr = 1
            else:
                hr = 0
            
            # 5. Calculate NDCG @ 10 (Normalized Discounted Cumulative Gain)
            # This rewards the model for putting the item at Rank 1 vs Rank 10.
            # Formula: 1 / log2(Rank + 1)
            if rank <= 10:
                ndcg = 1 / math.log2(rank + 1)
            else:
                ndcg = 0
            
            # 6. Calculate AUC (Area Under Curve)
            # This checks: "Is the True Item scored higher than the Decoys?"
            # Scikit-Learn handles the math here.
            try:
                auc = roc_auc_score(current_labels, scores)
            except ValueError:
                # Edge case: If all scores are exactly identical, AUC is undefined (0.5)
                auc = 0.5 
                
            return hr, ndcg, auc

        # --- Calculate for Our Model ---
        m_hr, m_ndcg, m_auc = calc_user_metrics(model_scores, labels)
        sum_hr_model += m_hr
        sum_ndcg_model += m_ndcg
        sum_auc_model += m_auc
        
        # --- Calculate for Popularity Baseline ---
        p_hr, p_ndcg, p_auc = calc_user_metrics(pop_scores, labels)
        sum_hr_pop += p_hr
        sum_ndcg_pop += p_ndcg
        sum_auc_pop += p_auc
        
        # Count this user
        total_users += 1

    # ============================================================
    # STEP D: Final Averages
    # ============================================================
    # Divide the running totals by the number of users to get the Average Metric.
    return {
        'Model': {
            'HR@10': sum_hr_model / total_users,
            'NDCG@10': sum_ndcg_model / total_users,
            'AUC': sum_auc_model / total_users
        },
        'Popularity': {
            'HR@10': sum_hr_pop / total_users,
            'NDCG@10': sum_ndcg_pop / total_users,
            'AUC': sum_auc_pop / total_users
        }
    }

# --- Run the Evaluation ---
# Using limit=1000 for testing purposes and bc my laptop is a cooker. For presentation, maybe use the entire thing?
results = evaluate_all_metrics(test_df, limit=1000)

print(f"\n--- FINAL RESULTS ---")
print(f"{'Metric':<10} | {'Popularity':<10} | {'Our Model':<10}")
print("-" * 35)
print(f"{'HR@10':<10} | {results['Popularity']['HR@10']:.4f}     | {results['Model']['HR@10']:.4f}")
print(f"{'NDCG@10':<10} | {results['Popularity']['NDCG@10']:.4f}     | {results['Model']['NDCG@10']:.4f}")
print(f"{'AUC':<10}   | {results['Popularity']['AUC']:.4f}     | {results['Model']['AUC']:.4f}")

# ==========================================
# 6. VISUALIZATION OF RESULTS
# ==========================================
import matplotlib.pyplot as plt

def plot_final_metrics(results):
    """
    Generates two plots:
    1. Bar Chart comparing Model vs Baseline on HR, NDCG, and AUC.
    2. Hit Rate Curve showing performance at different K values (1 to 20).
    """
    
    # --- PLOT 1: METRIC COMPARISON (BAR CHART) ---
    metrics = ['HR@10', 'NDCG@10', 'AUC']
    pop_vals = [results['Popularity'][m] for m in metrics]
    model_vals = [results['Model'][m] for m in metrics]
    
    x = np.arange(len(metrics))  # Label locations
    width = 0.35  # Bar width

    plt.figure(figsize=(14, 5))
    
    # Subplot 1: The Bar Chart
    ax1 = plt.subplot(1, 2, 1)
    rects1 = ax1.bar(x - width/2, pop_vals, width, label='Popularity', color='gray', alpha=0.7)
    rects2 = ax1.bar(x + width/2, model_vals, width, label='Our Model', color='#1f77b4')

    ax1.set_ylabel('Score')
    ax1.set_title('Baseline vs. Content Model Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, max(model_vals) * 1.2) # Add headroom for labels
    
    # Add text labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)

    # --- PLOT 2: HIT RATE CURVE (K=1 to 20) ---
    # To plot the curve, we need to quickly re-run the rank check for different K values.
    # We re-use the 'r_model' and 'r_pop' logic but in a simplified way for plotting.
    print("Generating Hit Rate Curve (this takes a moment)...")
    
    # Helper function to get raw ranks (needed for the curve)
    def get_raw_ranks(limit=500):
        ranks_m = []
        ranks_p = []
        subset = test_df.sample(n=limit, random_state=42)
        for _, row in subset.iterrows():
            u, true_item = row['u'], row['i']
            is_cold = u not in user_centroids
            
            candidates = [true_item]
            while len(candidates) < 100:
                decoy = random.choice(valid_items)
                if decoy != true_item: candidates.append(decoy)
            
            m_scores = []
            p_scores = []
            for item in candidates:
                if not is_cold:
                    vec = get_content_vector(u, item).reshape(1, -1)
                    m_scores.append(model.predict_proba(vec)[0][1])
                else:
                    m_scores.append(item_counts.get(item, 0))
                p_scores.append(item_counts.get(item, 0))
            
            # Get Rank for Model
            zipped_m = sorted(zip(m_scores, range(100)), key=lambda x: x[0], reverse=True)
            for rank, (score, idx) in enumerate(zipped_m):
                if idx == 0: # 0 is the index of true_item in candidates list
                    ranks_m.append(rank + 1)
                    break
            
            # Get Rank for Pop
            zipped_p = sorted(zip(p_scores, range(100)), key=lambda x: x[0], reverse=True)
            for rank, (score, idx) in enumerate(zipped_p):
                if idx == 0:
                    ranks_p.append(rank + 1)
                    break
        return ranks_m, ranks_p

    # Get ranks for 500 users to build the curve
    raw_ranks_model, raw_ranks_pop = get_raw_ranks(limit=500)
    
    k_values = range(1, 21)
    hr_curve_m = [sum(1 for r in raw_ranks_model if r <= k)/len(raw_ranks_model) for k in k_values]
    hr_curve_p = [sum(1 for r in raw_ranks_pop if r <= k)/len(raw_ranks_pop) for k in k_values]

    # Subplot 2: The Line Graph
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(k_values, hr_curve_m, marker='o', linewidth=2, label='Our Model')
    ax2.plot(k_values, hr_curve_p, marker='x', linestyle='--', color='gray', label='Popularity')
    
    ax2.set_xlabel('K (Top-K Recommendations)')
    ax2.set_ylabel('Hit Rate')
    ax2.set_title('Hit Rate @ K (Curve)')
    ax2.set_xticks([1, 5, 10, 15, 20])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Execute
plot_final_metrics(results)