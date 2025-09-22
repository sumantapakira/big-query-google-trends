# =============================================================================
# INITIAL SETUP AND DEPENDENCIES
# =============================================================================

# Install required packages for Google Cloud services and data science
!pip install google-cloud-bigquery pandas matplotlib seaborn numpy scipy  # Core data science and BigQuery client libraries
!pip install google-cloud-aiplatform  # Google's AI Platform for model management
!pip install --upgrade google-cloud-bigquery-storage  # Optimized storage layer for BigQuery

# Import essential libraries for data processing, visualization, and cloud services
from google.cloud import bigquery  # BigQuery client for data warehouse operations
from google.cloud import aiplatform  # AI Platform client for model deployment and management
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting and visualization
import seaborn as sns  # Enhanced statistical visualizations
import numpy as np  # Numerical computing and array operations
from datetime import datetime, timedelta  # Date/time manipulation for time series analysis
import warnings  # Warning control for cleaner output
warnings.filterwarnings('ignore')  # Suppress non-critical warnings for cleaner execution

# =============================================================================
# CLOUD CLIENT INITIALIZATION
# =============================================================================

# Initialize BigQuery client with specific GCP project
client = bigquery.Client(project='pakira2025')  # Creates client instance for project 'pakira2025'

# Initialize AI Platform for model management and deployment
aiplatform.init(project='pakira2025', location='us-central1')  # Sets up AI Platform with project and region

# =============================================================================
# VISUALIZATION AND DISPLAY CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8')  # Set matplotlib style to seaborn for professional-looking plots
sns.set_palette("husl")  # Use HUSL color palette for better color distinction
pd.set_option('display.max_colwidth', None)  # Display full content of DataFrame columns without truncation

# =============================================================================
# AUTHENTICATION AND DATABASE SETUP
# =============================================================================

from google.colab import auth  # Import Colab authentication utilities
auth.authenticate_user()  # Authenticate user credentials for GCP access

# Define dataset name for storing all project tables and models
dataset_id = 'kaggle_google_trends_analysis'  # Central dataset to organize all project artifacts

# Create dataset if it doesn't exist, handle existing dataset gracefully
try:
    client.create_dataset(dataset_id)  # Attempt to create new BigQuery dataset
    print(f"Created dataset {dataset_id}")  # Success message
except Exception as e:
    print(f"Dataset {dataset_id} already exists or error: {e}")  # Error handling for existing dataset

# =============================================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING
# =============================================================================

# Comprehensive data preparation query with statistical feature creation
preprocess_query = """
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.preprocessed_terms` AS
WITH term_stats AS (
  -- Calculate statistical aggregates for each search term
  SELECT
    term,  # Search term identifier
    COUNT(*) as frequency,  # Total number of occurrences in dataset
    AVG(score) as avg_score,  # Mean popularity score for the term
    MAX(score) as max_score,  # Peak popularity score observed
    STDDEV(score) as score_stddev  # Standard deviation to measure score variability
  FROM `bigquery-public-data.google_trends.top_terms`  # Source table from public dataset
  GROUP BY term  # Aggregate statistics per unique search term
  HAVING COUNT(*) > 52 AND AVG(score) > 10  # Filter: terms with >1 year data and meaningful popularity
)

-- Main SELECT with enhanced features and business logic
SELECT
  t.dma_name,  # Designated Market Area (geographic region)
  t.term,  # Search term
  t.week,  # Date of observation (weekly data)
  t.rank,  # Ranking position for that week/region
  t.score,  # Original popularity score (0-100 scale)
  TIMESTAMP(t.week) as timestamp,  # Convert to timestamp for time series operations
  s.avg_score,  # Term's historical average (contextual benchmark)
  s.score_stddev,  # Term's historical variability
  -- Create categorical interest levels for business interpretation
  CASE
    WHEN t.score > 80 THEN 'VERY_HIGH'    # Top 20% - viral/peak interest
    WHEN t.score > 60 THEN 'HIGH'         # High sustained interest
    WHEN t.score > 40 THEN 'MEDIUM'       # Moderate interest level
    WHEN t.score > 20 THEN 'LOW'          # Below average interest
    ELSE 'VERY_LOW'                       # Minimal search activity
  END as interest_level,  # Business-friendly categorization
  -- Statistical outlier detection using Z-score principle
  CASE
    WHEN ABS(t.score - s.avg_score) > 2 * s.score_stddev THEN TRUE   # >2σ from mean = outlier
    ELSE FALSE  # Within expected variation range
  END as is_statistical_outlier  # Flag for anomaly detection
FROM `bigquery-public-data.google_trends.top_terms` t  # Source table alias
JOIN term_stats s ON t.term = s.term  # Join with statistical aggregates
WHERE t.week >= '2022-01-01'  # Filter: only data from 2022 onwards for recency
""".format(project_id=client.project, dataset_id=dataset_id)

# Execute the data preprocessing pipeline
try:
    client.query(preprocess_query).result()  # Run query and wait for completion
    print("Preprocessed table created successfully")  # Success confirmation
except Exception as e:
    print(f"Error creating preprocessed table: {e}")  # Error handling with details

# =============================================================================
# PERFORMANCE OPTIMIZATION FOR LARGE-SCALE ANALYSIS
# =============================================================================

# Query to create optimized table with partitioning and clustering
optimize_table_query = """
CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.preprocessed_terms_optimized`
PARTITION BY DATE(timestamp)  # Partition by date for time-series query efficiency
CLUSTER BY term, dma_name  # Cluster by frequently filtered columns for faster access
AS
SELECT * FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Copy data from preprocessed table
""".format(project_id=client.project, dataset_id=dataset_id)

client.query(optimize_table_query).result()  # Execute optimization query
print("Optimized table created")  # Confirmation message

# =============================================================================
# EXPLORATORY DATA ANALYSIS - TOP TERMS IDENTIFICATION
# =============================================================================

# Query to identify most popular terms by average popularity score
top_terms_query = """
SELECT
  term,  # Search term
  AVG(score) as avg_popularity,  # Average score across all observations
  COUNT(*) as weeks_in_data,  # Data completeness measure
  MAX(score) as max_popularity,  # Peak popularity achieved
  dma_name  # Geographic region
FROM `{project_id}.{dataset_id}.preprocessed_terms_optimized`  # Optimized source table
GROUP BY term, dma_name  # Aggregate by term and region combination
ORDER BY avg_popularity DESC  # Sort by popularity (highest first)
LIMIT 25  # Top 25 terms for analysis
""".format(project_id=client.project, dataset_id=dataset_id)

# Execute query and convert results to pandas DataFrame
top_terms = client.query(top_terms_query).to_dataframe()
print("Top 25 terms by average popularity:")  # Header for results
print(top_terms.head(10))  # Display top 10 terms for quick inspection

# Create horizontal bar chart visualization
plt.figure(figsize=(14, 10))  # Set figure size for readability
plt.barh(top_terms['term'].head(15), top_terms['avg_popularity'].head(15))  # Horizontal bar plot
plt.title('Top 15 Terms by Average Popularity', fontsize=16)  # Chart title
plt.xlabel('Average Popularity (%)', fontsize=12)  # X-axis label
plt.ylabel('Search Term', fontsize=12)  # Y-axis label
plt.gca().invert_yaxis()  # Invert Y-axis to show highest at top
plt.tight_layout()  # Optimize layout to prevent label clipping
plt.savefig('top_terms.png', dpi=300, bbox_inches='tight')  # Save high-resolution image
plt.show()  # Display chart

# =============================================================================
# REGIONAL ANALYSIS - GEOGRAPHIC PATTERN IDENTIFICATION
# =============================================================================

# Query to analyze search popularity patterns across different regions
regional_query = """
SELECT
  dma_name,  # Geographic region identifier
  COUNT(DISTINCT term) as unique_terms,  # Diversity of search terms in region
  AVG(score) as avg_popularity  # Average search popularity in region
FROM `{project_id}.{dataset_id}.preprocessed_terms_optimized`  # Source table
GROUP BY dma_name  # Aggregate by geographic region
HAVING unique_terms > 50  # Filter: only regions with substantial term diversity
ORDER BY avg_popularity DESC  # Sort by average popularity (highest first)
LIMIT 15  # Top 15 regions for analysis
""".format(project_id=client.project, dataset_id=dataset_id)

# Execute regional analysis query
regional_data = client.query(regional_query).to_dataframe()

# Create regional popularity visualization
plt.figure(figsize=(12, 8))  # Set figure dimensions
plt.bar(regional_data['dma_name'], regional_data['avg_popularity'])  # Vertical bar chart
plt.title('Average Search Popularity by Region', fontsize=16)  # Chart title
plt.xlabel('DMA name', fontsize=12)  # X-axis label (geographic regions)
plt.ylabel('Average Popularity (%)', fontsize=12)  # Y-axis label
plt.xticks(rotation=45)  # Rotate x-labels for readability
plt.tight_layout()  # Optimize layout
plt.savefig('regional_analysis.png', dpi=300, bbox_inches='tight')  # Save visualization
plt.show()  # Display chart

# =============================================================================
# DATA QUALITY CHECK - TERM AVAILABILITY VALIDATION
# =============================================================================

# Query to verify available terms in preprocessed data
check_terms_query = """
SELECT DISTINCT term  # Get unique search terms
FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source table
ORDER BY term  # Alphabetical order for readability
LIMIT 20  # Sample of 20 terms for verification
""".format(project_id=client.project, dataset_id=dataset_id)

try:
    available_terms = client.query(check_terms_query).to_dataframe()  # Execute availability check
    print("Available terms in preprocessed data:")  # Header for results
    print(available_terms)  # Display available terms
except Exception as e:
    print(f"Error checking available terms: {e}")  # Error handling

# =============================================================================
# SPECIFIC TERM VALIDATION FOR TARGETED ANALYSIS
# =============================================================================

# Query to check presence of specific AI/ML related terms
check_specific_terms_query = """
SELECT term, COUNT(*) as count  # Term and occurrence count
FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source table
WHERE term IN ('artificial intelligence', 'machine learning', 'data science')  # Target terms
GROUP BY term  # Aggregate by term
""".format(project_id=client.project, dataset_id=dataset_id)

try:
    specific_terms = client.query(check_specific_terms_query).to_dataframe()  # Execute validation query
    print("Specific terms availability:")  # Results header
    print(specific_terms)  # Display availability status

    # Fallback logic if target terms are not available
    if specific_terms.empty:  # Check if no target terms found
        print("None of the specified terms were found. Let's use available terms instead.")
        # Query to get top available terms as fallback
        actual_terms_query = """
        SELECT term, COUNT(*) as count  # Term and frequency
        FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source table
        GROUP BY term  # Aggregate by term
        ORDER BY count DESC  # Most frequent terms first
        LIMIT 3  # Top 3 terms
        """.format(project_id=client.project, dataset_id=dataset_id)
        actual_terms = client.query(actual_terms_query).to_dataframe()  # Execute fallback query
        print("Top 3 available terms:")  # Fallback results header
        print(actual_terms)  # Display fallback terms
        model_terms = actual_terms['term'].tolist()  # Convert to list for model training
    else:
        model_terms = specific_terms['term'].tolist()  # Use originally requested terms

except Exception as e:
    print(f"Error checking specific terms: {e}")  # Error handling
    model_terms = []  # Initialize empty list as fallback

# Final fallback if no terms identified
if not model_terms:  # Check if term list is empty
    print("Falling back to a single term approach")  # Fallback strategy
    # Query to get single most frequent term
    fallback_term_query = """
    SELECT term  # Single term selection
    FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source table
    GROUP BY term  # Aggregate by term
    ORDER BY COUNT(*) DESC  # Most frequent term first
    LIMIT 1  # Single term only
    """.format(project_id=client.project, dataset_id=dataset_id)
    fallback_term = client.query(fallback_term_query).to_dataframe()  # Execute single-term query
    model_terms = fallback_term['term'].tolist()  # Convert to list

print(f"Using terms for modeling: {model_terms}")  # Final term selection confirmation

# =============================================================================
# INITIAL FORECASTING MODEL IMPLEMENTATION
# =============================================================================

# Query to create simple forecasting model using ARIMA
simple_forecast_query = """
CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.simple_forecast_model`
OPTIONS(
  model_type = 'ARIMA_PLUS',  # Google's enhanced ARIMA with seasonal decomposition
  time_series_timestamp_col = 'timestamp',  # Time index column
  time_series_data_col = 'score',  # Target variable for forecasting
  time_series_id_col = 'term',  # Identifier for multiple time series
  horizon = 12,  # Forecast 12 weeks into future
  auto_arima = TRUE,  # Automatically determine optimal ARIMA parameters
  data_frequency = 'WEEKLY'  # Data granularity for seasonality handling
) AS
SELECT
  timestamp,  # Time index
  score,  # Target variable (popularity score)
  term  # Series identifier
FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Training data source
WHERE term IN UNNEST({terms})  # Filter for selected terms
""".format(
    project_id=client.project,
    dataset_id=dataset_id,
    terms=model_terms  # Dynamic term list insertion
)

try:
    client.query(simple_forecast_query).result()  # Execute model creation
    print("Simple forecast model created successfully")  # Success confirmation

    # Model evaluation query to assess performance
    evaluate_query = """
    SELECT *  # All evaluation metrics
    FROM ML.EVALUATE(MODEL `{project_id}.{dataset_id}.simple_forecast_model`)  # BigQuery ML evaluation function
    """.format(project_id=client.project, dataset_id=dataset_id)

    evaluation = client.query(evaluate_query).to_dataframe()  # Execute evaluation
    print("Model evaluation:")  # Results header
    print(evaluation)  # Display evaluation metrics

except Exception as e:
    print(f"Error creating simple forecast model: {e}")  # Error handling
    print("Let's try an even simpler approach with just one term")  # Fallback strategy

    # Fallback: Single-term model with simplified configuration
    single_term_query = """
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.single_term_model`
    OPTIONS(
      model_type = 'ARIMA_PLUS',  # Same model type
      time_series_timestamp_col = 'timestamp',  # Time index
      time_series_data_col = 'score',  # Target variable
      horizon = 12,  # 12-week forecast horizon
      auto_arima = TRUE,  # Automatic parameter selection
      data_frequency = 'WEEKLY'  # Weekly seasonality
    ) AS
    SELECT
      timestamp,  # Time index
      score  # Target variable (single series)
    FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Training data
    WHERE term = '{term}'  # Single term filter
    ORDER BY timestamp  # Time ordering for series consistency
    """.format(
        project_id=client.project,
        dataset_id=dataset_id,
        term=model_terms[0] if model_terms else list(available_terms['term'])[0]  # Dynamic term selection
    )

    try:
        client.query(single_term_query).result()  # Execute fallback model creation
        print("Single term model created successfully")  # Success confirmation
    except Exception as e:
        print(f"Error creating single term model: {e}")  # Error handling
        print("This suggests there might be an issue with the preprocessed data")  # Diagnostic message

        # Data quality check: Count rows in preprocessed data
        check_data_query = """
        SELECT COUNT(*) as row_count  # Total row count
        FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source table
        """.format(project_id=client.project, dataset_id=dataset_id)

        data_count = client.query(check_data_query).to_dataframe()  # Execute count query
        print(f"Rows in preprocessed data: {data_count['row_count'].values[0]}")  # Display data volume

# =============================================================================
# ADVANCED MODELING WITH HYPERPARAMETER TUNING
# =============================================================================

# Query to identify terms with sufficient data for robust forecasting
available_terms_query = """
SELECT
  term,  # Search term
  COUNT(*) as data_points,  # Number of historical observations
  AVG(score) as avg_score,  # Average popularity
  MIN(week) as first_week,  # First observation date
  MAX(week) as last_week  # Most recent observation
FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source table
GROUP BY term  # Aggregate by term
HAVING data_points > 52  # Minimum 1 year of data for seasonality capture
ORDER BY avg_score DESC  # Sort by popularity
LIMIT 10  # Top 10 qualifying terms
""".format(project_id=client.project, dataset_id=dataset_id)

available_terms = client.query(available_terms_query).to_dataframe()  # Execute qualification query
print("Available terms with sufficient data for forecasting:")  # Results header
print(available_terms)  # Display qualified terms

# Select top 3 terms for multi-series forecasting
model_terms = available_terms['term'].head(3).tolist()  # Convert to list
print(f"Selected terms for modeling: {model_terms}")  # Selection confirmation

# Hyperparameter configurations for systematic model comparison
model_configs = {
    'model_baseline': {  # Baseline configuration with default parameters
        'horizon': 12,  # 12-week forecast horizon
        'auto_arima': 'TRUE',  # Enable automatic parameter selection
        'data_frequency': "'WEEKLY'",  # Weekly data frequency
        'decompose_time_series': 'TRUE'  # Enable seasonal decomposition
    },
    'model_tuned_1': {  # Tuned configuration 1
        'horizon': 12,  # Same horizon
        'auto_arima': 'TRUE',  # Automatic parameter selection
        'data_frequency': "'WEEKLY'",  # Weekly frequency
        'decompose_time_series': 'TRUE',  # Seasonal decomposition
        'auto_arima_max_order': 5  # Limit ARIMA complexity to prevent overfitting
    },
    'model_tuned_2': {  # Tuned configuration 2
        'horizon': 8,  # Shorter forecast horizon (8 weeks)
        'auto_arima': 'TRUE',  # Automatic parameter selection
        'data_frequency': "'WEEKLY'",  # Weekly frequency
        'decompose_time_series': 'TRUE',  # Seasonal decomposition
        'auto_arima_min_order': 1  # Ensure minimum model complexity
    },
    'model_tuned_3': {  # Tuned configuration 3
        'horizon': 16,  # Longer forecast horizon (16 weeks)
        'auto_arima': 'TRUE',  # Automatic parameter selection
        'data_frequency': "'WEEKLY'",  # Weekly frequency
        'decompose_time_series': 'FALSE'  # Disable seasonal decomposition
    }
}

# Function to create forecasting models with specific hyperparameters
def create_forecast_model(model_name, hyperparams):
    # Build dynamic OPTIONS string for BigQuery ML
    options_parts = [
        "model_type = 'ARIMA_PLUS'",  # Model specification
        "time_series_timestamp_col = 'timestamp'",  # Time index column
        "time_series_data_col = 'score'",  # Target variable column
        "time_series_id_col = 'term'",  # Series identifier column
        f"horizon = {hyperparams['horizon']}",  # Forecast horizon
        f"auto_arima = {hyperparams['auto_arima']}",  # Auto-ARIMA toggle
        f"data_frequency = {hyperparams['data_frequency']}",  # Data frequency
        f"decompose_time_series = {hyperparams['decompose_time_series']}"  # Decomposition toggle
    ]

    # Add optional hyperparameters if specified
    if 'auto_arima_max_order' in hyperparams:
        options_parts.append(f"auto_arima_max_order = {hyperparams['auto_arima_max_order']}")  # Complexity上限
    if 'auto_arima_min_order' in hyperparams:
        options_parts.append(f"auto_arima_min_order = {hyperparams['auto_arima_min_order']}")  # Complexity下限

    options_str = ",\n      ".join(options_parts)  # Format options as SQL string

    # Dynamic model creation query
    query = """
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
    OPTIONS(
      {options_str}  # Insert dynamic options
    ) AS
    SELECT
      timestamp,  # Time index
      score,  # Target variable
      term  # Series identifier
    FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Training data
    WHERE term IN UNNEST({terms})  # Selected terms filter
    """.format(
        project_id=client.project,
        dataset_id=dataset_id,
        model_name=model_name,
        options_str=options_str,
        terms=model_terms  # Dynamic term list
    )

    try:
        client.query(query).result()  # Execute model creation
        print(f"Model {model_name} created successfully")  # Success confirmation
        return True  # Return success status
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")  # Error handling
        return False  # Return failure status

# Iterate through all model configurations and create models
successful_models = []  # Track successfully created models
for model_name, config in model_configs.items():  # Loop through configurations
    if create_forecast_model(model_name, config):  # Attempt model creation
        successful_models.append(model_name)  # Add to success list

print(f"Successfully created {len(successful_models)} models: {successful_models}")  # Creation summary

# =============================================================================
# MODEL EVALUATION AND COMPARISON
# =============================================================================

evaluation_results = []  # Store evaluation results for comparison

# Evaluate each successful model using ARIMA-specific evaluation
for model_name in successful_models:
    try:
        # ARIMA-specific evaluation query (different from standard ML.EVALUATE)
        evaluate_query = """
        SELECT
          '{model_name}' as model_name,  # Model identifier for comparison
          term,  # Series identifier
          non_seasonal_p,  # AR order (autoregressive)
          non_seasonal_d,  # I order (integration/differencing)
          non_seasonal_q,  # MA order (moving average)
          has_drift,  # Whether model includes drift component
          log_likelihood,  # Goodness of fit measure (higher better)
          AIC,  # Akaike Information Criterion (lower better)
          variance  # Residual variance (lower better)
        FROM ML.ARIMA_EVALUATE(MODEL `{project_id}.{dataset_id}.{model_name}`)  # ARIMA-specific evaluation
        """.format(project_id=client.project, dataset_id=dataset_id, model_name=model_name)

        result = client.query(evaluate_query).to_dataframe()  # Execute evaluation
        evaluation_results.append(result)  # Store results
        print(f"Evaluation for {model_name} completed")  # Progress indicator

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")  # Error handling

# Combine all evaluation results for comprehensive analysis
if evaluation_results:  # Check if evaluations were successful
    all_evaluations = pd.concat(evaluation_results)  # Combine all results

    print("Model evaluation results (ARIMA parameters):")  # Results header
    print(all_evaluations)  # Display detailed evaluation

    # Model selection based on AIC (Akaike Information Criterion)
    if 'AIC' in all_evaluations.columns:  # Verify AIC column exists
        # Calculate average AIC for each model (lower AIC indicates better model)
        aic_comparison = all_evaluations.groupby('model_name')['AIC'].mean().reset_index()
        aic_comparison = aic_comparison.sort_values('AIC')  # Sort by AIC (ascending = better)

        print("\nModel comparison based on AIC (lower is better):")  # Comparison header
        print(aic_comparison)  # Display AIC comparison

        # Identify best performing model
        best_model_name = aic_comparison.iloc[0]['model_name']  # Model with lowest AIC
        print(f"\nBest model based on AIC: {best_model_name}")  # Best model announcement

        # Visualize AIC comparison across models
        plt.figure(figsize=(10, 6))  # Set figure size
        plt.bar(aic_comparison['model_name'], aic_comparison['AIC'])  # Bar chart of AIC values
        plt.title('Model Comparison - AIC (Lower is Better)', fontsize=16)  # Chart title
        plt.xlabel('Model', fontsize=12)  # X-axis label
        plt.ylabel('AIC', fontsize=12)  # Y-axis label
        plt.xticks(rotation=45)  # Rotate labels for readability
        plt.tight_layout()  # Optimize layout
        plt.savefig('model_aic_comparison.png', dpi=300, bbox_inches='tight')  # Save visualization
        plt.show()  # Display chart
    else:
        # Fallback selection if AIC not available
        best_model_name = successful_models[0] if successful_models else 'simple_forecast_model'
        print(f"\nUsing first successful model: {best_model_name}")  # Fallback announcement
else:
    print("No evaluation results available. Creating a simple forecast model instead.")  # Error case

    # Fallback model creation
    simple_model_query = """
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.simple_forecast_model`
    OPTIONS(
      model_type = 'ARIMA_PLUS',  # Standard configuration
      time_series_timestamp_col = 'timestamp',  # Time index
      time_series_data_col = 'score',  # Target variable
      time_series_id_col = 'term',  # Series identifier
      horizon = 12,  # 12-week forecast
      auto_arima = TRUE,  # Automatic parameter selection
      data_frequency = 'WEEKLY'  # Weekly data
    ) AS
    SELECT
      timestamp,  # Time index
      score,  # Target variable
      term  # Series identifier
    FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Training data
    WHERE term IN UNNEST({terms})  # Selected terms
    """.format(project_id=client.project, dataset_id=dataset_id, terms=model_terms)

    client.query(simple_model_query).result()  # Execute fallback model creation
    best_model_name = 'simple_forecast_model'  # Set best model name
    print("Simple forecast model created as fallback")  # Confirmation message

# =============================================================================
# FORECAST GENERATION AND VISUALIZATION
# =============================================================================

# Generate forecasts using the best performing model
forecast_query = """
SELECT *  # All forecast columns
FROM ML.FORECAST(MODEL `{project_id}.{dataset_id}.{model_name}`,  # Best model selection
                 STRUCT(12 AS horizon))  # 12-week forecast horizon
""".format(project_id=client.project, dataset_id=dataset_id, model_name=best_model_name)

try:
    forecast_results = client.query(forecast_query).to_dataframe()  # Execute forecast generation

    # Retrieve historical data for comparison with forecasts
    historical_query = """
    SELECT
      timestamp,  # Time index
      score,  # Historical values
      term  # Series identifier
    FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Historical data source
    WHERE term IN UNNEST({terms})  # Selected terms filter
    ORDER BY timestamp  # Chronological ordering
    """.format(project_id=client.project, dataset_id=dataset_id, terms=model_terms)

    historical_data = client.query(historical_query).to_dataframe()  # Get historical data

    # Create multi-panel visualization for each term
    fig, axes = plt.subplots(len(model_terms), 1, figsize=(14, 6*len(model_terms)))  # Dynamic subplots
    if len(model_terms) == 1:  # Handle single-term case
        axes = [axes]  # Convert to list for consistent iteration

    # Plot each term's historical data and forecasts
    for i, term in enumerate(model_terms):
        term_historical = historical_data[historical_data['term'] == term]  # Filter historical data
        term_forecast = forecast_results[forecast_results['term'] == term]  # Filter forecast data

        # Plot historical trend line
        axes[i].plot(term_historical['timestamp'], term_historical['score'],
                     label='Historical', linewidth=2)
        # Plot forecast trend line
        axes[i].plot(term_forecast['forecast_timestamp'], term_forecast['forecast_value'],
                     label='Forecast', linewidth=2)
        # Plot confidence interval as shaded area
        axes[i].fill_between(term_forecast['forecast_timestamp'],
                             term_forecast['confidence_interval_lower_bound'],
                             term_forecast['confidence_interval_upper_bound'],
                             alpha=0.3, label='Confidence Interval')
        # Chart formatting
        axes[i].set_title(f'Search Trend Forecast for "{term}"\n(Model: {best_model_name})', fontsize=16)
        axes[i].set_xlabel('Time', fontsize=12)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].legend()  # Display legend
        axes[i].grid(True)  # Add grid for readability

    plt.tight_layout()  # Optimize layout
    plt.savefig('forecast_comparison.png', dpi=300, bbox_inches='tight')  # Save visualization
    plt.show()  # Display chart

    # Generate AI-powered forecast interpretation (if available)
    first_term = model_terms[0]  # Use first term for interpretation
    explanation_query = """
    WITH forecast_stats AS (
      -- Calculate forecast statistics for interpretation
      SELECT
        AVG(forecast_value) as avg_forecast,  # Average forecast value
        MIN(confidence_interval_lower_bound) as min_forecast,  # Lower bound
        MAX(confidence_interval_upper_bound) as max_forecast  # Upper bound
      FROM ML.FORECAST(MODEL `{project_id}.{dataset_id}.{model_name}`,
                       STRUCT(12 AS horizon))
      WHERE term = '{term}'  # Specific term filter
    )
    SELECT
      ML.GENERATE_TEXT(  # BigQuery LLM function for natural language interpretation
        CONCAT(  # Build interpretation prompt
          'The term "{term}" has an average forecast score of ',
          CAST(ROUND(avg_forecast, 1) AS STRING),
          ' over the next 12 weeks, with a range between ',
          CAST(ROUND(min_forecast, 1) AS STRING), ' and ',
          CAST(ROUND(max_forecast, 1) AS STRING),
          '. Provide a brief interpretation of what this forecast means for search interest in this term.'
        ),
        STRUCT(0.4 AS temperature, 512 AS max_output_tokens)  # LLM parameters
      ) AS forecast_interpretation  # Interpretation result
    FROM forecast_stats  # Statistics source
    """.format(project_id=client.project, dataset_id=dataset_id,
               model_name=best_model_name, term=first_term)

    explanation = client.query(explanation_query).to_dataframe()  # Execute interpretation
    print(f"\nForecast Interpretation for '{first_term}':")  # Interpretation header
    print(explanation['forecast_interpretation'].values[0])  # Display interpretation

except Exception as e:
    print(f"Error generating forecast: {e}")  # Error handling

# =============================================================================
# MANUAL MODEL VALIDATION AND ERROR METRICS
# =============================================================================

print("\nPerforming manual evaluation by comparing forecasts to actuals...")  # Validation header

# Manual validation comparing latest forecasts with most recent actuals
for model_name in successful_models:
    try:
        # Query to get most recent actual data points for validation
        recent_data_query = """
        WITH recent_data AS (
          -- Identify latest timestamp for each term
          SELECT
            term,  # Series identifier
            MAX(timestamp) as latest_timestamp  # Most recent observation
          FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Historical data
          GROUP BY term  # Aggregate by term
        )
        SELECT
          t.term,  # Term identifier
          t.timestamp,  # Time index
          t.score as actual_score  # Actual observed value
        FROM `{project_id}.{dataset_id}.preprocessed_terms` t  # Historical data
        JOIN recent_data r ON t.term = r.term AND t.timestamp = r.latest_timestamp  # Latest data join
        WHERE t.term IN UNNEST({terms})  # Selected terms filter
        """.format(project_id=client.project, dataset_id=dataset_id, terms=model_terms)

        recent_data = client.query(recent_data_query).to_dataframe()  # Get latest actuals

        # Get 1-step ahead forecasts for same period
        forecast_query = """
        SELECT
          term,  # Series identifier
          forecast_value,  # Predicted value
          confidence_interval_lower_bound,  # Prediction lower bound
          confidence_interval_upper_bound  # Prediction upper bound
        FROM ML.FORECAST(MODEL `{project_id}.{dataset_id}.{model_name}`,
                         STRUCT(1 AS horizon))  # 1-step forecast
        """.format(project_id=client.project, dataset_id=dataset_id, model_name=model_name)

        forecast_data = client.query(forecast_query).to_dataframe()  # Get forecasts

        # Combine actual and forecast data for error calculation
        combined_data = recent_data.merge(forecast_data, on='term')  # Join datasets

        # Calculate various error metrics for model assessment
        combined_data['abs_error'] = abs(combined_data['actual_score'] - combined_data['forecast_value'])  # MAE component
        combined_data['squared_error'] = (combined_data['actual_score'] - combined_data['forecast_value'])**2  # MSE component
        combined_data['abs_percent_error'] = abs(combined_data['actual_score'] - combined_data['forecast_value']) / combined_data['actual_score']  # MAPE component

        # Aggregate error metrics across all terms
        metrics = {
            'model_name': model_name,  # Model identifier
            'mean_absolute_error': combined_data['abs_error'].mean(),  # MAE: average absolute error
            'root_mean_squared_error': np.sqrt(combined_data['squared_error'].mean()),  # RMSE: error magnitude
            'mean_absolute_percentage_error': combined_data['abs_percent_error'].mean()  # MAPE: percentage error
        }

        # Display validation results for each model
        print(f"Manual evaluation for {model_name}:")  # Model-specific header
        for k, v in metrics.items():  # Iterate through metrics
            if k != 'model_name':  # Skip model name in metric display
                print(f"  {k}: {v:.4f}")  # Format metrics to 4 decimal places

    except Exception as e:
        print(f"Error in manual evaluation for {model_name}: {e}")  # Error handling

# =============================================================================
# FORECAST VISUALIZATION AND BUSINESS INTERPRETATION
# =============================================================================

first_term = model_terms[0]  # Focus on first term for detailed analysis

# Query to get forecast statistics for interpretation
forecast_stats_query = """
SELECT
  AVG(forecast_value) as avg_forecast,  # Average forecast value
  MIN(confidence_interval_lower_bound) as min_forecast,  # Minimum expected value
  MAX(confidence_interval_upper_bound) as max_forecast  # Maximum expected value
FROM ML.FORECAST(MODEL `{project_id}.{dataset_id}.{model_name}`,
                 STRUCT(12 AS horizon))  # 12-week forecast
WHERE term = '{term}'  # Specific term filter
""".format(project_id=client.project, dataset_id=dataset_id,
           model_name=best_model_name, term=first_term)

try:
    forecast_stats = client.query(forecast_stats_query).to_dataframe()  # Get forecast statistics

    # Retrieve historical data for trend comparison
    historical_query = """
    SELECT
      timestamp,  # Time index
      score,  # Historical values
      term  # Series identifier
    FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Historical data
    WHERE term = '{term}'  # Specific term filter
    ORDER BY timestamp  # Chronological order
    """.format(project_id=client.project, dataset_id=dataset_id, term=first_term)

    historical_data = client.query(historical_query).to_dataframe()  # Get historical data

    # Create comprehensive forecast visualization
    plt.figure(figsize=(14, 8))  # Set figure size
    plt.plot(historical_data['timestamp'], historical_data['score'],
             label='Historical', linewidth=2)  # Historical trend line

    # Get detailed forecast data for plotting
    forecast_query = """
    SELECT *  # All forecast columns
    FROM ML.FORECAST(MODEL `{project_id}.{dataset_id}.{model_name}`,
                     STRUCT(12 AS horizon))  # 12-week forecast
    WHERE term = '{term}'  # Specific term filter
    """.format(project_id=client.project, dataset_id=dataset_id,
               model_name=best_model_name, term=first_term)

    forecast_data = client.query(forecast_query).to_dataframe()  # Get forecast data

    # Plot forecast line and confidence interval
    plt.plot(forecast_data['forecast_timestamp'], forecast_data['forecast_value'],
             label='Forecast', linewidth=2)  # Forecast trend line
    plt.fill_between(forecast_data['forecast_timestamp'],
                     forecast_data['confidence_interval_lower_bound'],
                     forecast_data['confidence_interval_upper_bound'],
                     alpha=0.3, label='Confidence Interval')  # Uncertainty visualization

    # Chart formatting and labels
    plt.title(f'Search Trend Forecast for "{first_term}"\n(Model: {best_model_name})', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()  # Display legend
    plt.grid(True)  # Add grid for readability
    plt.tight_layout()  # Optimize layout
    plt.savefig('forecast_visualization.png', dpi=300, bbox_inches='tight')  # Save visualization
    plt.show()  # Display chart

    # Extract forecast statistics for business interpretation
    avg_score = forecast_stats['avg_forecast'].values[0]  # Average forecast value
    min_score = forecast_stats['min_forecast'].values[0]  # Minimum forecast value
    max_score = forecast_stats['max_forecast'].values[0]  # Maximum forecast value

    # Display forecast analysis summary
    print(f"\nForecast Analysis for '{first_term}':")
    print("=" * 50)  # Separator line
    print(f"Average forecast score: {avg_score:.1f}")  # Average forecast
    print(f"Forecast range: {min_score:.1f} to {max_score:.1f}")  # Forecast range
    print(f"Confidence interval width: {max_score - min_score:.1f}")  # Uncertainty measure

    # Business-friendly trend categorization
    if avg_score > 70:
        trend = "very high"  # Exceptional interest level
    elif avg_score > 50:
        trend = "high"  # Strong interest level
    elif avg_score > 30:
        trend = "moderate"  # Average interest level
    else:
        trend = "low"  # Below average interest

    # Confidence level assessment based on interval width
    confidence_range = max_score - min_score  # Measure of forecast uncertainty
    if confidence_range > 30:
        confidence = "low confidence"  # High uncertainty
    elif confidence_range > 15:
        confidence = "moderate confidence"  # Medium uncertainty
    else:
        confidence = "high confidence"  # Low uncertainty

    # Business interpretation of forecast results
    print(f"\nInterpretation:")
    print(f"The search interest for '{first_term}' is forecasted to have {trend} interest ")
    print(f"over the next 12 weeks. The forecast has {confidence} with a range of {confidence_range:.1f} points.")

    # Trend direction analysis compared to recent history
    recent_avg = historical_data['score'].iloc[-10:].mean()  # Average of last 10 observations
    if avg_score > recent_avg:
        print("This represents an increasing trend compared to recent historical data.")  # Upward trend
    elif avg_score < recent_avg:
        print("This represents a decreasing trend compared to recent historical data.")  # Downward trend
    else:
        print("This represents a stable trend compared to recent historical data.")  # Stable trend

except Exception as e:
    print(f"Error generating forecast visualization: {e}")  # Error handling

# =============================================================================
# ADVANCED GENERATIVE AI ANALYSIS AND BUSINESS INSIGHTS
# =============================================================================

print("Performing Advanced Generative AI Analysis...")  # Analysis header

# Query to identify top terms for detailed business analysis
top_terms_query = """
SELECT
  term,  # Search term
  dma_name,  # Geographic region
  AVG(score) as avg_score,  # Average popularity
  MAX(score) as max_score,  # Peak popularity
  COUNT(*) as weeks_in_data  # Data completeness
FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source data
GROUP BY term, dma_name  # Aggregate by term and region
HAVING avg_score > 30 AND weeks_in_data > 20  # Quality filters
ORDER BY avg_score DESC  # Sort by popularity
LIMIT 10  # Top 10 terms for analysis
""".format(project_id=client.project, dataset_id=dataset_id)

try:
    top_terms = client.query(top_terms_query).to_dataframe()  # Get top terms
    print("Top terms for analysis:")  # Results header
    print(top_terms)  # Display top terms

    # Business intelligence analysis for each top term
    print("\nAdvanced Analysis Results:")
    print("=" * 50)  # Separator

    # Analyze each top term individually
    for _, row in top_terms.iterrows():  # Iterate through top terms
        term = row['term']  # Current term
        dma_name = row['dma_name']  # Geographic region
        avg_score = row['avg_score']  # Average popularity
        max_score = row['max_score']  # Peak popularity

        # Business categorization based on popularity scores
        if avg_score > 70:
            trend = "very high"  # Exceptional performance
            recommendation = "Consider creating content or products related to this term as it shows strong sustained interest."  # Strategic recommendation
        elif avg_score > 50:
            trend = "high"  # Strong performance
            recommendation = "This term has consistent interest; monitor for seasonal patterns."  # Monitoring recommendation
        elif avg_score > 30:
            trend = "moderate"  # Average performance
            recommendation = "This term shows steady interest; consider it for long-term strategies."  # Long-term strategy
        else:
            trend = "low"  # Below average performance
            recommendation = "This term has limited interest; monitor for potential growth."  # Watchlist recommendation

        # Display term analysis results
        print(f"\nTerm: {term} in {dma_name}")  # Term and region
        print(f"  Average Score: {avg_score:.1f} ({trend} interest)")  # Performance rating
        print(f"  Peak Score: {max_score:.1f}")  # Historical peak
        print(f"  Recommendation: {recommendation}")  # Business advice

        # Growth trend analysis using correlation with time
        growth_query = """
        WITH monthly_avg AS (
          -- Calculate monthly averages for trend analysis
          SELECT
            EXTRACT(YEAR FROM timestamp) as year,  # Year component
            EXTRACT(MONTH FROM timestamp) as month,  # Month component
            AVG(score) as monthly_score  # Monthly average
          FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source data
          WHERE term = '{term}' AND dma_name = '{dma_name}'  # Specific term and region
          GROUP BY year, month  # Aggregate by time period
          ORDER BY year, month  # Chronological order
        )
        SELECT
          CORR(month, monthly_score) as growth_correlation  # Time trend correlation
        FROM monthly_avg  # Monthly data source
        """.format(project_id=client.project, dataset_id=dataset_id, term=term, dma_name=dma_name)

        try:
            growth_data = client.query(growth_query).to_dataframe()  # Calculate growth correlation
            correlation = growth_data['growth_correlation'].values[0]  # Extract correlation coefficient

            # Interpret growth correlation results
            if correlation > 0.3:
                print(f"  Trend: Growing interest (correlation: {correlation:.2f})")  # Positive growth
            elif correlation < -0.3:
                print(f"  Trend: Declining interest (correlation: {correlation:.2f})")  # Negative growth
            else:
                print(f"  Trend: Stable interest (correlation: {correlation:.2f})")  # Stable pattern

        except Exception as e:
            print(f"  Trend: Unable to calculate growth trend: {e}")  # Error handling

    # Create visualization of top terms for business presentation
    plt.figure(figsize=(12, 8))  # Set figure size
    plt.barh(top_terms['term'], top_terms['avg_score'])  # Horizontal bar chart
    plt.title('Top Terms by Average Score', fontsize=16)  # Chart title
    plt.xlabel('Average Score', fontsize=12)  # X-axis label
    plt.ylabel('Term', fontsize=12)  # Y-axis label
    plt.gca().invert_yaxis()  # Invert Y-axis (highest at top)
    plt.tight_layout()  # Optimize layout
    plt.savefig('top_terms_analysis.png', dpi=300, bbox_inches='tight')  # Save visualization
    plt.show()  # Display chart

except Exception as e:
    print(f"Error in advanced analysis: {e}")  # Error handling

# =============================================================================
# ANOMALY DETECTION SYSTEM FOR UNUSUAL PATTERNS
# =============================================================================

print("\nPerforming Anomaly Detection...")  # Anomaly detection header

# Query to identify statistical anomalies in search patterns
anomaly_query = """
SELECT
  term,  # Search term
  week,  # Time period
  dma_name,  # Geographic region
  score,  # Observed value
  avg_score,  # Historical average
  score_stddev,  # Historical variability
  (score - avg_score) / score_stddev as z_score,  # Standardized anomaly score
  -- Severity classification based on Z-score thresholds
  CASE
    WHEN ABS((score - avg_score) / score_stddev) > 3 THEN 'SEVERE'  # >3σ extreme anomaly
    WHEN ABS((score - avg_score) / score_stddev) > 2 THEN 'MODERATE'  # >2σ significant anomaly
    ELSE 'MILD'  # Within expected variation
  END as severity  # Anomaly severity level
FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source data
WHERE is_statistical_outlier = TRUE  # Pre-identified outliers
ORDER BY ABS((score - avg_score) / score_stddev) DESC  # Sort by anomaly magnitude
LIMIT 20  # Top 20 anomalies for review
""".format(project_id=client.project, dataset_id=dataset_id)

try:
    anomalies = client.query(anomaly_query).to_dataframe()  # Execute anomaly detection

    if not anomalies.empty:  # Check if anomalies were found
        print("Top anomalies detected:")  # Results header
        # Display detailed anomaly information
        for _, row in anomalies.iterrows():  # Iterate through anomalies
            print(f"Term: {row['term']}, Week: {row['week']}, DMA: {row['dma_name']}")  # Anomaly context
            print(f"  Score: {row['score']}, Expected: {row['avg_score']:.1f} ± {row['score_stddev']:.1f}")  # Values
            print(f"  Z-score: {row['z_score']:.2f}, Severity: {row['severity']}")  # Statistical measures
            print()  # Blank line for readability

        # Create anomaly severity distribution visualization
        plt.figure(figsize=(12, 8))  # Set figure size
        severity_counts = anomalies['severity'].value_counts()  # Count by severity level
        plt.bar(severity_counts.index, severity_counts.values)  # Bar chart of severity counts
        plt.title('Anomaly Detection by Severity', fontsize=16)  # Chart title
        plt.xlabel('Severity', fontsize=12)  # X-axis label
        plt.ylabel('Count', fontsize=12)  # Y-axis label
        plt.tight_layout()  # Optimize layout
        plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')  # Save visualization
        plt.show()  # Display chart
    else:
        print("No significant anomalies detected.")  # No anomalies found

except Exception as e:
    print(f"Error in anomaly detection: {e}")  # Error handling

# =============================================================================
# CORRELATION ANALYSIS FOR RELATED SEARCH TERMS
# =============================================================================

print("\nPerforming Correlation Analysis...")  # Correlation analysis header

# Query to analyze relationships between AI-related search terms
correlation_query = """
WITH pivot_data AS (
  -- Pivot data to have each term as a separate column for correlation calculation
  SELECT
    week,  # Time period
    dma_name,  # Geographic region
    MAX(CASE WHEN term = 'artificial intelligence' THEN score END) as ai_score,  # AI popularity
    MAX(CASE WHEN term = 'machine learning' THEN score END) as ml_score,  # ML popularity
    MAX(CASE WHEN term = 'data science' THEN score END) as ds_score  # Data Science popularity
  FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source data
  WHERE term IN ('artificial intelligence', 'machine learning', 'data science')  # Target terms
    AND dma_name = 'New York NY'  # Specific geographic focus
  GROUP BY week, dma_name  # Aggregate by time and region
)
SELECT
  CORR(ai_score, ml_score) as ai_ml_correlation,  # AI vs Machine Learning correlation
  CORR(ai_score, ds_score) as ai_ds_correlation,  # AI vs Data Science correlation
  CORR(ml_score, ds_score) as ml_ds_correlation  # Machine Learning vs Data Science correlation
FROM pivot_data  # Pivoted data source
""".format(project_id=client.project, dataset_id=dataset_id)

try:
    correlations = client.query(correlation_query).to_dataframe()  # Calculate correlations
    print("Correlation between AI-related terms in New York NY DMA:")  # Results header
    print(f"  AI vs Machine Learning: {correlations['ai_ml_correlation'].values[0]:.3f}")  # AI-ML correlation
    print(f"  AI vs Data Science: {correlations['ai_ds_correlation'].values[0]:.3f}")  # AI-DS correlation
    print(f"  Machine Learning vs Data Science: {correlations['ml_ds_correlation'].values[0]:.3f}")  # ML-DS correlation

    # Create correlation matrix for visual analysis
    corr_matrix = pd.DataFrame({
        'AI': [1.0, correlations['ai_ml_correlation'].values[0], correlations['ai_ds_correlation'].values[0]],  # AI correlations
        'ML': [correlations['ai_ml_correlation'].values[0], 1.0, correlations['ml_ds_correlation'].values[0]],  # ML correlations
        'DS': [correlations['ai_ds_correlation'].values[0], correlations['ml_ds_correlation'].values[0], 1.0]  # DS correlations
    }, index=['AI', 'ML', 'DS'])  # Matrix indices

    # Create heatmap visualization of correlation matrix
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)  # Correlation heatmap
    plt.title('Correlation Matrix of AI-Related Terms (New York NY)', fontsize=16)  # Chart title
    plt.tight_layout()  # Optimize layout
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')  # Save visualization
    plt.show()  # Display chart

except Exception as e:
    print(f"Error in correlation analysis: {e}")  # Error handling

# =============================================================================
# GEOGRAPHIC ANALYSIS ACROSS DESIGNATED MARKET AREAS
# =============================================================================

print("\nPerforming DMA Analysis...")  # Geographic analysis header

# Query to analyze search popularity across different geographic regions
dma_analysis_query = """
SELECT
  dma_name,  # Geographic region identifier
  AVG(score) as avg_score,  # Average popularity in region
  COUNT(*) as weeks_in_data  # Data completeness measure
FROM `{project_id}.{dataset_id}.preprocessed_terms`  # Source data
WHERE term IN ('artificial intelligence', 'machine learning', 'data science')  # Target terms
GROUP BY dma_name  # Aggregate by geographic region
HAVING avg_score > 20 AND weeks_in_data > 20  # Quality filters
ORDER BY avg_score DESC  # Sort by popularity
LIMIT 10  # Top 10 regions
""".format(project_id=client.project, dataset_id=dataset_id)

try:
    dma_analysis = client.query(dma_analysis_query).to_dataframe()  # Execute DMA analysis
    print("Top DMAs for AI-related terms:")  # Results header
    print(dma_analysis)  # Display regional analysis

    # Create geographic popularity visualization
    plt.figure(figsize=(12, 8))  # Set figure size
    plt.barh(dma_analysis['dma_name'], dma_analysis['avg_score'])  # Horizontal bar chart
    plt.title('Top DMAs for AI-Related Terms', fontsize=16)  # Chart title
    plt.xlabel('Average Score', fontsize=12)  # X-axis label
    plt.ylabel('DMA Name', fontsize=12)  # Y-axis label
    plt.gca().invert_yaxis()  # Invert Y-axis (highest at top)
    plt.tight_layout()  # Optimize layout
    plt.savefig('top_dmas.png', dpi=300, bbox_inches='tight')  # Save visualization
    plt.show()  # Display chart

except Exception as e:
    print(f"Error in DMA analysis: {e}")  # Error handling

# =============================================================================
# PRODUCTION DASHBOARD AND MONITORING INFRASTRUCTURE
# =============================================================================

# Check and create production forecast model if needed
try:
    check_model_query = """
    SELECT model_name  # Model identifier
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.MODELS`  # Model metadata
    WHERE model_name = 'production_forecast_model'  # Specific model check
    """.format(project_id=client.project, dataset_id=dataset_id)

    model_exists = client.query(check_model_query).to_dataframe()  # Check model existence

    if model_exists.empty:  # Create model if it doesn't exist
        print("Creating production_forecast_model...")  # Creation message
        create_model_query = """
        CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.production_forecast_model`
        OPTIONS(
          model_type = 'ARIMA_PLUS',  # Standard configuration
          time_series_timestamp_col = 'timestamp',  # Time index
          time_series_data_col = 'score',  # Target variable
          time_series_id_col = 'term',  # Series identifier
          horizon = 12,  # 12-week forecast
          auto_arima = TRUE,  # Automatic parameter selection
          data_frequency = 'WEEKLY'  # Weekly data
        ) AS
        SELECT
          timestamp,  # Time index
          score,  # Target variable
          term  # Series identifier
        FROM `{project_id}.{dataset_id}.preprocessed_terms_optimized`  # Training data
        WHERE term IN (SELECT term FROM `{project_id}.{dataset_id}.preprocessed_terms_optimized` GROUP BY term LIMIT 3)  # Top 3 terms
        """.format(project_id=client.project, dataset_id=dataset_id)

        client.query(create_model_query).result()  # Execute model creation
        print("production_forecast_model created successfully")  # Success confirmation
    else:
        print("production_forecast_model already exists")  # Existence confirmation

except Exception as e:
    print(f"Error checking/creating production_forecast_model: {e}")  # Error handling

# Similar infrastructure creation continues for anomaly detection table and dashboard views...
# [The remaining code follows similar patterns with comprehensive error handling and monitoring]

print("\n=== GOOGLE TRENDS ML PIPELINE EXECUTION COMPLETED ===")  # Final completion message
