import json
import re
from pathlib import Path

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas

from spkatt_gepade.common import matching_precision_recall_f1

plt.rcParams["font.size"] = "7"
plt.rcParams["mathtext.rm"] = "Open Sans"
plt.rcParams["mathtext.it"] = "Open Sans:italic"
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# NOTE: install dependencies with poetry install --with=dev

GOLD_PATH = Path('./data/dev/task1')
PRED_PATH = Path('./predictions_dev')


def get_dataframe():
    gold_annotations = {}
    pred_annotations = {}

    for f in GOLD_PATH.glob('*.json'):
        gold_annotations[f.name] = json.load(open(f))['Annotations']

    for f in PRED_PATH.glob('*.json'):
        pred_annotations[f.name] = json.load(open(f))['Annotations']

    precision_obs, recall_obs = matching_precision_recall_f1(gold_annotations, pred_annotations, aggregated=False)
    group_count = pandas.Series([re.search(r'CDUCSU|SPD|AfD|FDP|GRUENE|LINKE', f).group() for (f, _, _), _, _, _
                                   in precision_obs + recall_obs]).value_counts().to_dict()
    unique_groups = list(group_count.keys())

    n_success, n_trials, group_id, pr_or_rec = [], [], [], []
    for kind, observations in zip([0, 1], [precision_obs, recall_obs]):
        n_success.extend([k for _, _, k, n in observations])
        n_trials.extend([n for _, _, k, n in observations])
        group_id.extend(
            [unique_groups.index(re.search(r'CDUCSU|SPD|AfD|FDP|GRUENE|LINKE', f).group()) for (f, _, _), _, _, _
             in observations])
        pr_or_rec.extend([kind] * len(observations))

    df = pandas.DataFrame()
    df['n_success'] = np.array(n_success)
    df['n_trials'] = np.array(n_trials)
    df['group_id'] = np.array(group_id)
    df['pr_or_rec'] = np.array(pr_or_rec)
    return df, group_count


print('==== calculating scores')
dataframe, group_count = get_dataframe()
unique_groups = list(group_count.keys())

colormap = {'CDUCSU': 'black', 'GRUENE': 'green', 'AfD': '#3279e3', 'SPD': 'red', 'LINKE': 'purple',
            'FDP': '#c79200'}

def make_pooled_model(df):
    with pm.Model(coords={'group': unique_groups, 'metric': ['precision', 'recall']}) as model_pooled:
        mu = pm.Beta('mu', alpha=1.5, beta=1.5, dims=('group', 'metric'))
        nu = pm.Exponential('nu', 1, dims=('group', 'metric'))

        alpha = pm.Deterministic('alpha', mu * nu, dims=('group', 'metric'))
        beta = pm.Deterministic('beta', (1-mu) * nu, dims=('group', 'metric'))

        outcome = pm.BetaBinomial('outcome',
                                  alpha=alpha[df['group_id'], df['pr_or_rec']],
                                  beta=beta[df['group_id'], df['pr_or_rec']],
                                  n=df['n_trials'],
                                  observed=df['n_success'])
        f1 = pm.Deterministic('f1', 2 / ((1/mu[:,0]) + 1/mu[:,1]), dims=('group'))

    return model_pooled

def make_unpooled_model(df):
    with pm.Model(coords={'group': unique_groups, 'metric': ['precision', 'recall']}) as model_unpooled:
        mu = pm.Beta('mu', alpha=1.5, beta=1.5, dims=('metric'))
        nu = pm.Exponential('nu', 1, dims=('metric'))

        alpha = pm.Deterministic('alpha', mu * nu, dims=('metric'))
        beta = pm.Deterministic('beta', (1 - mu) * nu, dims=('metric'))

        outcome = pm.BetaBinomial('outcome',
                                  alpha=alpha[df['pr_or_rec']],
                                  beta=beta[df['pr_or_rec']],
                                  n=df['n_trials'],
                                  observed=df['n_success'])
        f1 = pm.Deterministic('f1', 2 / ((1/mu[0]) + 1/mu[1]))

    return model_unpooled

# prior predictive check!

# fig, ax = plt.subplots(figsize=(3, 1.5), dpi=300)
# with make_pooled_model():
#     prior_idata = pm.sample_prior_predictive(20)
#     prior_pred = prior_idata.prior.stack(sample=('chain', 'draw')).sel(group='CDUCSU')
#     for a, b in zip(prior_pred['alpha'].data.reshape(-1),prior_pred['beta'].data.reshape(-1)):
#         x = np.linspace(betadist.ppf(0.01, a, b), betadist.ppf(0.99, a, b), 100)
#         ax.plot(x, betadist.pdf(x, a, b), color='black', lw=0.4)
#
# ax.set_ylim((0, 4))
# ax.set_ylabel('density')
# plt.tight_layout()
# plt.savefig('figures/prior_predictive_check.pdf')
# plt.show()


print('==== make pooled inferences')
with make_pooled_model(dataframe):
    idata = pm.sample()


f1_post = pandas.DataFrame({'n': group_count})
f1_post['f1'] = 0

fig, ax = plt.subplots(figsize=(5.9, 1.7), dpi=150)
for group in unique_groups:
    post_ = idata.posterior['f1'].sel(group=group)
    f1_post.loc[group, 'f1'] = post_.stack(sample=('chain', 'draw')).values.mean()

    for chain in post_.chain:
        grid, pdf = az.kde(post_.sel(chain=chain).values)
        ax.plot(grid, pdf, color=colormap[group], lw=0.5)
    ax.plot([0.8],[0.8], color=colormap[group], lw=3, label=f'{group} ({group_count[group]})')

ax.legend()
ax.set_title('(a) Estimated posterior distributions for F1 scores; pooled model')
ax.set_ylabel('density')
plt.tight_layout()
plt.show()

print(f1_post.to_string())


# f1 = idata.posterior['f1'].stack(sample=('chain', 'draw'))
# diff = f1.max(dim='group') - f1.min(dim='group')
# print('pr(diff > 0.05) = ', (diff > 0.05).values.mean())
#
# print('pr(SPD and LINKE in ROPE) = ', ((f1.sel(group='LINKE').mean() - f1.sel(group='SPD')) < 0.05).mean().values)
# print('E(SPD) = ', ((f1.sel(group='SPD').mean()).mean().values))
# print('E(LINKE) = ', ((f1.sel(group='LINKE').mean()).mean().values))


print('==== make unpooled inferences')

with make_unpooled_model(dataframe):
    idata = pm.sample()


fig, ax = plt.subplots(figsize=(5.9, 1.3), dpi=150)
post_ = idata.posterior['f1']
for chain in post_.chain:
    grid, pdf = az.kde(post_.sel(chain=chain).values)
    ax.plot(grid, pdf, color='black', lw=0.5)

ax.set_title('(b) Estimated posterior distribution for F1 score; unpooled model')
ax.set_ylabel('density')
plt.tight_layout()
plt.show()

f1_post = pandas.DataFrame({'n': {'overall': len(dataframe)}, 'f1': {'overall': post_.stack(sample=('chain', 'draw')).values.mean()}})
print(f1_post.to_string())


print('==== perform model comparison')

def make_pooled_model_binom(df):
    # alternative models
    with pm.Model(coords={'group': unique_groups, 'metric': ['precision', 'recall']}) as model_pooled:
        p = pm.Beta('p', alpha=1, beta=1, dims=('group', 'metric'))

        outcome = pm.Binomial('outcome',
                              p=p[df['group_id'], df['pr_or_rec']],
                              n=df['n_trials'],
                              observed=df['n_success'])
        f1 = pm.Deterministic('f1', 2 / ((1/p[:,0]) + 1/p[:,1]), dims=('group'))
        # prior_idata = pm.sample_prior_predictive(20)
        # idata = pm.sample(idata_kwargs={"log_likelihood": True})

    return model_pooled

def make_unpooled_model_binom(df):
    with pm.Model(coords={'group': unique_groups, 'metric': ['precision', 'recall']}) as model_unpooled:
        p = pm.Beta('p', alpha=1, beta=1, dims=('metric'))

        outcome = pm.Binomial('outcome',
                              p=p[df['pr_or_rec']],
                              n=df['n_trials'],
                              observed=df['n_success'])
        f1 = pm.Deterministic('f1', 2 / ((1/p[0]) + 1/p[1]))

    return model_unpooled

with make_unpooled_model(dataframe):
    unpooled_idata = pm.sample(idata_kwargs={"log_likelihood": True})
with make_pooled_model(dataframe):
    pooled_idata = pm.sample(idata_kwargs={"log_likelihood": True})
with make_unpooled_model_binom(dataframe):
    binom_unpooled_idata = pm.sample(idata_kwargs={"log_likelihood": True})
with make_pooled_model_binom(dataframe):
    binom_pooled_idata = pm.sample(idata_kwargs={"log_likelihood": True})


df_comp_loo = az.compare({'unpooled betabinom': unpooled_idata, 'pooled betabinom': pooled_idata, 'pooled binom': binom_pooled_idata, 'unpooled binom': binom_unpooled_idata})
print(df_comp_loo.to_string())
az.plot_compare(df_comp_loo, figsize=(12, 6))
plt.tight_layout()
plt.show()