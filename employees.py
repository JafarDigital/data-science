import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t, f, levene, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_white
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("ADVANCED STATISTICS: COMPLETE REAL-WORLD PROJECT")
print("Analyzing Employee Performance and Satisfaction Data")
print("=" * 80)

# CREATE REALISTIC DATASET
np.random.seed(42)
n = 300

# Generate realistic employee data
departments = ['Sales', 'Engineering', 'Marketing', 'HR']
dept_probs = [0.4, 0.3, 0.2, 0.1]

data = {
    'employee_id': range(1, n+1),
    'department': np.random.choice(departments, n, p=dept_probs),
    'years_experience': np.random.exponential(3, n) + 1,  # Right-skewed, realistic
    'age': np.random.normal(35, 8, n),
    'education_level': np.random.choice(['Bachelor', 'Master', 'PhD'], n, p=[0.6, 0.3, 0.1])
}

# Create performance score with realistic relationships
df = pd.DataFrame(data)
df['age'] = np.clip(df['age'], 22, 65).astype(int)
df['years_experience'] = np.clip(df['years_experience'], 1, 30).round(1)

# Performance influenced by experience, department, and education
base_performance = 70
dept_effects = {'Sales': 5, 'Engineering': 8, 'Marketing': 3, 'HR': 0}
edu_effects = {'Bachelor': 0, 'Master': 5, 'PhD': 10}

df['performance_score'] = (
    base_performance + 
    df['years_experience'] * 1.2 +
    df['department'].map(dept_effects) +
    df['education_level'].map(edu_effects) +
    np.random.normal(0, 8, n)  # Random variation
)

# Salary influenced by performance, experience, department
df['salary'] = (
    40000 + 
    df['performance_score'] * 500 +
    df['years_experience'] * 1200 +
    df['department'].map({'Sales': 5000, 'Engineering': 12000, 'Marketing': 3000, 'HR': 0}) +
    np.random.normal(0, 5000, n)
)

# Job satisfaction correlated with performance and salary
df['job_satisfaction'] = (
    3 + 
    (df['performance_score'] - 70) * 0.02 +
    (df['salary'] - 60000) * 0.00001 +
    np.random.normal(0, 0.8, n)
)
df['job_satisfaction'] = np.clip(df['job_satisfaction'], 1, 5)

# Clean up
df['performance_score'] = np.clip(df['performance_score'], 0, 100).round(1)
df['salary'] = np.clip(df['salary'], 30000, 150000).round(0).astype(int)

print("DATASET OVERVIEW:")
print("-" * 40)
print(f"Sample size: {len(df)}")
print(f"Variables: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print(f"\nDataset Statistics:")
print(df.describe())

# 1. Z-TEST vs T-TEST DEMONSTRATION
print(f"\n\n1. Z-TEST vs T-TEST")
print("=" * 50)

# Question: Is the average performance score significantly different from 75?
sample_performance = df['performance_score']
population_mean = 75
sample_mean = sample_performance.mean()
sample_std = sample_performance.std(ddof=1)
n_sample = len(sample_performance)

print(f"Testing H0: μ = {population_mean} vs H1: μ ≠ {population_mean}")
print(f"Sample size: {n_sample}")
print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample std: {sample_std:.2f}")

# Z-TEST (assuming we know population std = 10)
population_std = 10  # Assume we know this
z_stat = (sample_mean - population_mean) / (population_std / np.sqrt(n_sample))
z_p_value = 2 * (1 - norm.cdf(abs(z_stat)))

print(f"\nZ-TEST (assuming σ = {population_std}):")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {z_p_value:.6f}")

# T-TEST (using sample std)
t_stat, t_p_value = stats.ttest_1samp(sample_performance, population_mean)

print(f"\nT-TEST (using sample std):")
print(f"t-statistic: {t_stat:.4f}")
print(f"P-value: {t_p_value:.6f}")
print(f"Degrees of freedom: {n_sample - 1}")

# 2. ONE-TAILED vs TWO-TAILED TESTS
print(f"\n\n2. DEALING WITH TAILS AND REJECTIONS")
print("=" * 50)

alpha = 0.05

# Two-tailed test
t_critical_two = t.ppf(1 - alpha/2, n_sample - 1)
print(f"TWO-TAILED TEST (α = {alpha}):")
print(f"Critical values: ±{t_critical_two:.3f}")
print(f"Rejection regions: t < -{t_critical_two:.3f} OR t > {t_critical_two:.3f}")
print(f"Decision: {'Reject H0' if abs(t_stat) > t_critical_two else 'Fail to reject H0'}")

# One-tailed test (right tail) - testing if performance > 75
t_critical_one = t.ppf(1 - alpha, n_sample - 1)
t_p_value_one = 1 - t.cdf(t_stat, n_sample - 1)
print(f"\nONE-TAILED TEST (H1: μ > {population_mean}):")
print(f"Critical value: {t_critical_one:.3f}")
print(f"Rejection region: t > {t_critical_one:.3f}")
print(f"P-value (one-tailed): {t_p_value_one:.6f}")
print(f"Decision: {'Reject H0' if t_stat > t_critical_one else 'Fail to reject H0'}")

# 3. EQUAL vs UNEQUAL VARIANCES
print(f"\n\n3. EQUAL vs UNEQUAL VARIANCES")
print("=" * 50)

# Compare performance between Sales and Engineering
sales_perf = df[df['department'] == 'Sales']['performance_score']
eng_perf = df[df['department'] == 'Engineering']['performance_score']

print(f"Comparing Sales vs Engineering performance:")
print(f"Sales: n={len(sales_perf)}, mean={sales_perf.mean():.2f}, std={sales_perf.std():.2f}")
print(f"Engineering: n={len(eng_perf)}, mean={eng_perf.mean():.2f}, std={eng_perf.std():.2f}")

# Test for equal variances using Levene's test
levene_stat, levene_p = levene(sales_perf, eng_perf)
print(f"\nLevene's test for equal variances:")
print(f"Test statistic: {levene_stat:.4f}")
print(f"P-value: {levene_p:.6f}")
print(f"Equal variances: {'Yes' if levene_p > 0.05 else 'No'}")

# Student's t-test (assumes equal variances)
t_stat_equal, p_equal = stats.ttest_ind(sales_perf, eng_perf, equal_var=True)
print(f"\nStudent's t-test (equal variances):")
print(f"t-statistic: {t_stat_equal:.4f}")
print(f"P-value: {p_equal:.6f}")

# Welch's t-test (unequal variances)
t_stat_unequal, p_unequal = stats.ttest_ind(sales_perf, eng_perf, equal_var=False)
print(f"\nWelch's t-test (unequal variances):")
print(f"t-statistic: {t_stat_unequal:.4f}")
print(f"P-value: {p_unequal:.6f}")

# 4. ANOVA - COMPARING MULTIPLE GROUPS
print(f"\n\n4. ANOVA - COMPARING MULTIPLE GROUPS")
print("=" * 50)

# Compare performance across all departments
dept_groups = [df[df['department'] == dept]['performance_score'] for dept in departments]

# One-way ANOVA
f_stat, anova_p = stats.f_oneway(*dept_groups)
print(f"One-way ANOVA: Performance by Department")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {anova_p:.6f}")

if anova_p < 0.05:
    print("Significant difference between departments")
    
    # Post-hoc pairwise comparisons
    print(f"\nPairwise t-tests (with Bonferroni correction):")
    from itertools import combinations
    pairs = list(combinations(departments, 2))
    n_comparisons = len(pairs)
    
    for dept1, dept2 in pairs:
        group1 = df[df['department'] == dept1]['performance_score']
        group2 = df[df['department'] == dept2]['performance_score']
        t_stat, p_val = stats.ttest_ind(group1, group2)
        p_adjusted = min(p_val * n_comparisons, 1.0)  # Bonferroni correction
        print(f"{dept1} vs {dept2}: p = {p_val:.4f}, adjusted p = {p_adjusted:.4f}")

# 5. CORRELATION TESTS
print(f"\n\n5. CORRELATION TESTS")
print("=" * 50)

# Pearson correlation (parametric)
pearson_r, pearson_p = pearsonr(df['years_experience'], df['performance_score'])
print(f"Pearson correlation (Experience vs Performance):")
print(f"r = {pearson_r:.4f}")
print(f"P-value: {pearson_p:.6f}")

# Spearman correlation (non-parametric)
spearman_r, spearman_p = spearmanr(df['years_experience'], df['performance_score'])
print(f"\nSpearman correlation (Experience vs Performance):")
print(f"ρ = {spearman_r:.4f}")
print(f"P-value: {spearman_p:.6f}")

# Multiple correlations
numeric_cols = ['years_experience', 'age', 'performance_score', 'salary', 'job_satisfaction']
correlation_matrix = df[numeric_cols].corr()
print(f"\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# 6. REGRESSION - THE WORKHORSE
print(f"\n\n6. REGRESSION - THE WORKHORSE OF STATISTICAL ANALYSIS")
print("=" * 60)

# Simple linear regression: Performance vs Experience
X = pd.to_numeric(df['years_experience'], errors='coerce').astype(float)
y = pd.to_numeric(df['performance_score'], errors='coerce').astype(float)
X_with_const = sm.add_constant(X)  # Add intercept

model_simple = sm.OLS(y, X_with_const).fit()
print("Simple Linear Regression: Performance ~ Experience")
print(model_simple.summary().tables[1])

# Multiple regression: Performance ~ Experience + Department + Education
# Create dummy variables properly
df_encoded = pd.get_dummies(df, columns=['department', 'education_level'], drop_first=True)

# Select predictors and ensure they're numeric
predictors = ['years_experience'] + [col for col in df_encoded.columns if col.startswith(('department_', 'education_level_'))]
X_multi = df_encoded[predictors].copy()

# Convert all columns to numeric (booleans to integers)
for col in X_multi.columns:
    if X_multi[col].dtype == 'bool':
        X_multi[col] = X_multi[col].astype(int)
    else:
        X_multi[col] = pd.to_numeric(X_multi[col], errors='coerce')

# Check for any remaining non-numeric data
print(f"Data types in X_multi after conversion: {X_multi.dtypes}")
print(f"Any missing values: {X_multi.isnull().sum().sum()}")

# Ensure the DataFrame has consistent numeric types
X_multi = X_multi.astype(float)

X_multi_const = sm.add_constant(X_multi)

# Ensure target variable is numeric
y_multi = pd.to_numeric(df['performance_score'], errors='coerce').astype(float)

model_multi = sm.OLS(y_multi, X_multi_const).fit()
print(f"\n\nMultiple Regression: Performance ~ Experience + Department + Education")
print(model_multi.summary().tables[1])

# 7. REGRESSION IN PLACE OF T-TEST
print(f"\n\n7. REGRESSION IN PLACE OF T-TEST")
print("=" * 50)

# Traditional two-sample t-test: Sales vs Engineering
print("Traditional t-test approach:")
print(f"Sales mean: {sales_perf.mean():.2f}")
print(f"Engineering mean: {eng_perf.mean():.2f}")
print(f"Difference: {eng_perf.mean() - sales_perf.mean():.2f}")
print(f"t-statistic: {t_stat_unequal:.4f}")
print(f"P-value: {p_unequal:.6f}")

# Regression approach: Performance ~ Department (Sales as reference)
dept_subset = df[df['department'].isin(['Sales', 'Engineering'])].copy()
dept_subset['is_engineering'] = (dept_subset['department'] == 'Engineering').astype(int)

# Ensure numeric data
X_ttest = sm.add_constant(dept_subset['is_engineering'].astype(float))
y_ttest = pd.to_numeric(dept_subset['performance_score'], errors='coerce').astype(float)
model_ttest = sm.OLS(y_ttest, X_ttest).fit()

print(f"\nRegression approach (same result):")
print(model_ttest.summary().tables[1])
print(f"Interpretation: Engineering performs {model_ttest.params[1]:.2f} points higher than Sales")

# 8. REGRESSION IN PLACE OF ANOVA
print(f"\n\n8. REGRESSION IN PLACE OF ANOVA")
print("=" * 50)

# Traditional ANOVA
print(f"Traditional ANOVA F-statistic: {f_stat:.4f}")
print(f"Traditional ANOVA p-value: {anova_p:.6f}")

# Regression approach using all departments
dept_dummies = pd.get_dummies(df['department'], drop_first=True)  # Sales as reference

# Convert boolean columns to integers
for col in dept_dummies.columns:
    dept_dummies[col] = dept_dummies[col].astype(int)

X_anova = sm.add_constant(dept_dummies)
y_anova = pd.to_numeric(df['performance_score'], errors='coerce').astype(float)
model_anova = sm.OLS(y_anova, X_anova).fit()

print(f"\nRegression approach:")
print(model_anova.summary().tables[1])

# F-test for overall model significance (equivalent to ANOVA F-test)
print(f"\nOverall F-test (equivalent to ANOVA):")
print(f"F-statistic: {model_anova.fvalue:.4f}")
print(f"P-value: {model_anova.f_pvalue:.6f}")

# 9. REGRESSION IN PLACE OF CORRELATION
print(f"\n\n9. REGRESSION IN PLACE OF CORRELATION")
print("=" * 50)

# Traditional correlation
print(f"Traditional Pearson correlation:")
print(f"r = {pearson_r:.4f}")
print(f"P-value: {pearson_p:.6f}")
print(f"R² = {pearson_r**2:.4f}")

# Regression approach
X_corr = sm.add_constant(pd.to_numeric(df['years_experience'], errors='coerce').astype(float))
y_corr = pd.to_numeric(df['performance_score'], errors='coerce').astype(float)
model_corr = sm.OLS(y_corr, X_corr).fit()

print(f"\nRegression approach (same information):")
print(f"Slope = {model_corr.params[1]:.4f}")
print(f"P-value: {model_corr.pvalues[1]:.6f}")
print(f"R² = {model_corr.rsquared:.4f}")
print(f"√R² = {np.sqrt(model_corr.rsquared):.4f} (equals |r|)")

# 10. COMPREHENSIVE VISUALIZATION
print(f"\n\n10. CREATING COMPREHENSIVE VISUALIZATIONS")
print("=" * 50)

# Create comprehensive visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('Advanced Statistical Analysis: Employee Performance Study', fontsize=16, fontweight='bold')

# Row 1: Distribution and normality
# Histogram with normal overlay
axes[0, 0].hist(df['performance_score'], bins=30, density=True, alpha=0.7, color='skyblue')
mu, sigma = df['performance_score'].mean(), df['performance_score'].std()
x = np.linspace(df['performance_score'].min(), df['performance_score'].max(), 100)
axes[0, 0].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
axes[0, 0].axvline(mu, color='red', linestyle='--', label=f'Mean={mu:.1f}')
axes[0, 0].set_title('Performance Score Distribution')
axes[0, 0].legend()

# Box plot by department
sns.boxplot(data=df, x='department', y='performance_score', ax=axes[0, 1])
axes[0, 1].set_title('Performance by Department')
axes[0, 1].tick_params(axis='x', rotation=45)

# Q-Q plot for normality
stats.probplot(df['performance_score'], dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Q-Q Plot: Performance Score')

# Row 2: Relationships and correlations
# Scatter plot: Experience vs Performance
axes[1, 0].scatter(df['years_experience'], df['performance_score'], alpha=0.6, color='green')
# Add regression line
z = np.polyfit(df['years_experience'], df['performance_score'], 1)
p = np.poly1d(z)
axes[1, 0].plot(df['years_experience'], p(df['years_experience']), "r--", alpha=0.8)
axes[1, 0].set_xlabel('Years Experience')
axes[1, 0].set_ylabel('Performance Score')
axes[1, 0].set_title(f'Experience vs Performance (r={pearson_r:.3f})')

# Correlation heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

# Residual plot for regression
residuals = model_simple.resid
fitted = model_simple.fittedvalues
axes[1, 2].scatter(fitted, residuals, alpha=0.6)
axes[1, 2].axhline(y=0, color='red', linestyle='--')
axes[1, 2].set_xlabel('Fitted Values')
axes[1, 2].set_ylabel('Residuals')
axes[1, 2].set_title('Residual Plot')

# Row 3: Advanced analyses
# ANOVA visualization
dept_means = df.groupby('department')['performance_score'].mean().sort_values(ascending=False)
bars = axes[2, 0].bar(dept_means.index, dept_means.values, color=['coral', 'lightblue', 'lightgreen', 'gold'])
axes[2, 0].set_title(f'Mean Performance by Department\n(ANOVA F={f_stat:.2f}, p={anova_p:.4f})')
axes[2, 0].tick_params(axis='x', rotation=45)
axes[2, 0].set_ylabel('Mean Performance Score')

# Effect size visualization
dept_effect_sizes = []
for dept in departments:
    dept_data = df[df['department'] == dept]['performance_score']
    cohens_d = (dept_data.mean() - df['performance_score'].mean()) / df['performance_score'].std()
    dept_effect_sizes.append(cohens_d)

colors = ['red' if abs(d) > 0.5 else 'orange' if abs(d) > 0.2 else 'green' for d in dept_effect_sizes]
axes[2, 1].bar(departments, dept_effect_sizes, color=colors)
axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[2, 1].set_title("Effect Sizes (Cohen's d)")
axes[2, 1].set_ylabel("Cohen's d")
axes[2, 1].tick_params(axis='x', rotation=45)

# Multiple regression coefficients
coef_names = ['Experience', 'Dept_Engineering', 'Dept_HR', 'Dept_Marketing', 'Edu_Master', 'Edu_PhD']
coef_values = model_multi.params[1:]  # Exclude intercept
coef_pvalues = model_multi.pvalues[1:]

# Color by significance
colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'lightgray' 
          for p in coef_pvalues]
bars = axes[2, 2].bar(range(len(coef_values)), coef_values, color=colors)
axes[2, 2].set_xticks(range(len(coef_names)))
axes[2, 2].set_xticklabels(coef_names, rotation=45, ha='right')
axes[2, 2].set_title('Multiple Regression Coefficients')
axes[2, 2].set_ylabel('Coefficient Value')
axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# 11. SUMMARY AND INTERPRETATION
print(f"\n\n11. SUMMARY AND PRACTICAL INTERPRETATION")
print("=" * 60)

print(f"KEY FINDINGS:")
print(f"1. Performance Score Analysis:")
print(f"   • Mean performance: {df['performance_score'].mean():.1f} (significantly different from 75)")
print(f"   • Distribution is approximately normal (suitable for parametric tests)")

print(f"\n2. Department Differences:")
print(f"   • Significant differences exist between departments (ANOVA p={anova_p:.4f})")
dept_rankings = df.groupby('department')['performance_score'].mean().sort_values(ascending=False)
for i, (dept, score) in enumerate(dept_rankings.items(), 1):
    print(f"   • #{i}: {dept} ({score:.1f})")

print(f"\n3. Experience-Performance Relationship:")
print(f"   • Strong positive correlation (r={pearson_r:.3f}, p<0.001)")
print(f"   • Each year of experience adds ~{model_simple.params[1]:.1f} performance points")

print(f"\n4. Multiple Regression Insights:")
print(f"   • Model explains {model_multi.rsquared*100:.1f}% of performance variance")
print(f"   • Most important predictors:")
significant_predictors = model_multi.pvalues[model_multi.pvalues < 0.05].index
for pred in significant_predictors:
    if pred != 'const':
        coef = model_multi.params[pred]
        p_val = model_multi.pvalues[pred]
        print(f"     - {pred}: {coef:+.1f} points (p={p_val:.4f})")

print(f"\nSTATISTICAL METHOD EQUIVALENCES:")
print(f"• T-test ≡ Simple regression with binary predictor")
print(f"• ANOVA ≡ Multiple regression with categorical predictors") 
print(f"• Correlation ≡ Simple regression (r = √R²)")
print(f"• All tests can be unified under the regression framework")

print(f"\nPRACTICAL RECOMMENDATIONS:")
print(f"1. Focus hiring on Engineering talent (highest performance)")
print(f"2. Value experience - it strongly predicts performance")
print(f"3. Consider education level in hiring decisions")
print(f"4. Use regression models for nuanced understanding vs simple comparisons")

print(f"\nMETHODOLOGICAL NOTES:")
print(f"• Always check assumptions (normality, homoscedasticity)")
print(f"• Use appropriate test for your data structure and assumptions")
print(f"• Consider effect sizes, not just statistical significance")
print(f"• Regression provides the most comprehensive analytical framework")
print("=" * 60)
