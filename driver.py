### Last updated on January 30, 2022 by Thomas J. Nicoletti

# Import Python libraries
import easygui
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import string
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import chi2
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define Mahalanobis distance
def mahalanobis(x = None, data = None, cov = None):
	mu = x - np.mean(data)
	if not cov:
		cov = np.cov(data.values.T)
	inv = np.linalg.inv(cov)
	left = np.dot(mu, inv)
	mah = np.dot(left, mu.T)
	return mah.diagonal()

# Print user instructions
print('\n' + 'User Instructions' + '\n')
print('Dummy code or remove categorical features from your dataset before running this script.')
print('Ensure the outcome is located in the very last column of your dataset.')
print('This script will automatically detect whether your outcome is binary or multiclass.')
print('Replace any missing data in your dataset with the value NaN before running this script.' + '\n')

# Set random seed
seed = int(input('Please enter your numerical random seed for replication purposes: '))
np.random.seed(seed)
text = open('random_seed.txt', 'w')
text.write(f'The random seed for your analysis was {seed}.')
text.close()

# Describe the dataset
print('\n' + 'Please attach your dataset using the file explorer.')
df = pd.read_csv(easygui.fileopenbox())
df.dropna(inplace = True)
df['Mahalanobis'] = mahalanobis(x = df, data = df.iloc[:, :len(df.columns)], cov = None)
df['PValue'] = 1 - chi2.cdf(df['Mahalanobis'], len(df.columns) - 1)
out = int(input('\n' + 'Would you like to remove multivariate outliers, if applicable?' + '\n\n' +
	'0: No, do not remove multivariate outliers.' + '\n' + '1: Yes, remove multivariate outliers.' + '\n\n'))
if out == 1:
	df.drop(df[df.PValue < 0.0010].index, inplace = True)
df.drop(columns = ['Mahalanobis', 'PValue'], axis = 1, inplace = True)
n = df.shape[0]
text = open('sample_size.txt', 'w')
if out == 1:
	text.write(f'The sample size for your analysis was N = {n} after attempting to remove missing data and/or multivariate outliers.')
else:
	text.write(f'The sample size for your analysis was N = {n} after attempting to remove missing data.')
text.close()
x = df.iloc[:, :-1]
y = np.ravel(df.iloc[:, -1])
feat = df.columns[:-1]
mean = np.array(df.describe().loc['mean'][:-1])

# Check statistical assumptions
if len(feat) >= 2:
	bart = calculate_bartlett_sphericity(x)
	bart = (str(round(bart[1], 2)))
	text = open('sphericity.txt', 'w')
	if float(bart) < 0.01:
		text.write('Bartlett\'s Test of Sphericity (p < 0.01) indicated the observed correlation matrix was not an identity matrix.')
	elif float(bart) <= 0.05:
		text.write(f'Bartlett\'s Test of Sphericity (p = {bart}) indicated the observed correlation matrix was not an identity matrix.')
	else:
		text.write(f'Bartlett\'s Test of Sphericity (p = {bart}) indicated the observed correlation matrix was an identity matrix.')
	text.close()
	kmo = calculate_kmo(x)
	kmo = (str(round(kmo[1], 2)))
	text = open('factorability.txt', 'w')
	if float(kmo) < 0.01:
		text.write(f'The Kaiser-Meyer-Olkin Test (KMO < 0.01) indicated the data was not suitable for factor analysis.')
	elif float(kmo) <= 0.50:
		text.write(f'The Kaiser-Meyer-Olkin Test (KMO = {kmo}) indicated the data was not suitable for factor analysis.')
	else:
		text.write(f'The Kaiser-Meyer-Olkin Test (KMO = {kmo}) indicated the data was suitable for factor analysis.')
	text.close()

# Generate scree plot
if len(feat) >= 2:
	pca = PCA()
	pca.fit(x)
	comp = np.arange(pca.n_components_)
	plt.figure()
	plt.plot(comp, pca.explained_variance_ratio_, linewidth = 2, color = 'rosybrown', marker = 'o')
	ax = plt.gca()
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.title('Scree Plot')
	plt.xlabel('Principal Component')
	plt.ylabel('Proportion of Variance Explained →')
	plt.xticks(np.arange(0, len(comp), 1))
	plt.savefig('scree_plot.png', bbox_inches = 'tight')

# Request factor inputs
red = 0
factor = len(feat)
if len(feat) >= 2:
	print('\n' + 'Please review Bartlett\'s Test of Sphericity, the Kaiser-Meyer-Olkin Test, and the scree plot before proceeding.')
	red = int(input('\n' + 'Would you like to run an exploratory factor analysis before random forest classification?' + '\n\n' +
		'0: No, only run random forest classification.' + '\n' + '1: Yes, run exploratory factor analysis before random forest classification.' + '\n\n'))
	if red == 1:
		ncol = x.shape[1]
		factor = int(input('\n' + 'Please enter your desired number of factors for extraction: '))
		if factor >= ncol:
			factor = ncol - 1
			print('\n' + f'This value is too large; therefore, factors for extraction will equal the total number of features minus one (i.e., {factor}).')
		if factor <= 0:
			factor = 1
			print('\n' + 'This value must be a non-zero positive integer; therefore, factors for extraction will equal one.')

# Retrieve factor solution
if red == 1:
	if factor == 1:
		fa = FactorAnalysis(n_components = factor, max_iter = 3000)
	else:
		fa = FactorAnalysis(n_components = factor, max_iter = 3000, rotation = 'varimax')
	fa.fit(x)
	x = fa.transform(x)
	f = 'Factor '
	abc = list(string.ascii_uppercase)
	abc = ['{}{}'.format(f, i) for i in abc]
	ncol = x.shape[1]
	cols = []
	for i in range(0, ncol):
		cols.append(abc[i])
	load = pd.DataFrame(fa.components_.T.round(2), columns = cols, index = feat)
	load.to_csv('factor_loadings.csv')
	feat = cols
	x = pd.DataFrame(x, columns = cols)
	print('\n' + 'Please review the factor loadings before proceeding.')
	spec = int(input('\n' + 'Would you like to run the driver analysis with all extracted factors or with just the highest loading variable per factor?' + '\n\n' +
		'0: Run the analysis with all extracted factors.' + '\n' + '1: Run the analysis with the highest loading variable per factor.' + '\n\n'))
	if spec == 1:
		load = abs(load)
		keep = load.idxmax(axis = 0)
		keep.drop_duplicates(keep = 'first', inplace = True)
		keep = list(keep)
		df = df[df.columns.intersection(keep)]
		x = df
		feat = df.columns
		factor = len(feat)

# Run core analyses
if factor >= 2:
	vif = pd.Series(variance_inflation_factor(x.values, i) for i in range(x.shape[1]))
	vif = pd.DataFrame(np.array(vif.round(2)), columns = ['Variable Inflation Factor'], index = feat)
	vif.T.to_csv('variable_inflation_factors.csv')
clf = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_features = 'auto', bootstrap = True, oob_score = True, class_weight = 'balanced').fit(x, y)
oob = str(round(clf.oob_score_, 2)).ljust(4, '0')
pred = clf.predict_proba(x)
loss = str(round(log_loss(y, pred), 2)).ljust(4, '0')
perf = pd.DataFrame({'Out-of-Bag Score': oob, 'Log Loss': loss}, index = ['Estimate'])
perf.to_csv('model_performance.csv')

# Estimate feature importance
if factor >= 2:
	imp = clf.feature_importances_
	sort = np.argsort(imp)
	plt.figure()
	plt.barh(range(len(sort)), imp[sort], color = 'mediumaquamarine', align = 'center')
	plt.title('Feature Importance Plot')
	plt.xlabel('Derived Importance →')
	if max(imp[sort]) >= 1:
		limit = 0.01
	elif max(imp[sort]) >= 0.90:
		limit = 0.10
	else:
		limit = 0.20
	plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.xticks(np.arange(0, max(imp[sort]) + limit, 0.10))
	plt.yticks(range(len(sort)), [feat[i] for i in sort])
	for i, feat in enumerate(imp[sort]):
		plt.text(feat, i, str(round(feat, 2)).ljust(4, '0'))
	plt.savefig('feature_importance_plot.png', bbox_inches = 'tight')

# Plot quadrant chart
if factor >= 2:
	imps = []
	score = []
	for i, feat in enumerate(imp[sort]):
		imps.append(round(feat / imp[sort].mean() * 100, 0))
	for i, feat in enumerate(mean[sort]):
		score.append(round(feat / mean[sort].mean() * 100, 0))
	quad = pd.DataFrame({'Rescaled Observed Score →': score, 'Rescaled Derived Importance →': imps, 'Feature': x.columns[sort]})
	plt.figure()
	plt.subplots_adjust(top = 0.95)
	sns.lmplot(x = 'Rescaled Observed Score →', y = 'Rescaled Derived Importance →', data = quad, hue = 'Feature', legend = True, palette = 'Set2')
	ax = plt.gca()
	ax.set_title('Feature Quadrant Chart')
	ax.axvline(100, color = 'darkgray', linestyle = ':')
	ax.axhline(100, color = 'darkgray', linestyle = ':')
	plt.savefig('feature_quadrant_chart.png', bbox_inches = 'tight')
