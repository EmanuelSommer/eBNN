# %%
import bayesmates as bm
from bayesmates.config.data import Task
import bayesmates.inference.metrics as bmetrics
import plotnine as pn
import jax.numpy as jnp
import pandas as pd

# %%
predsaver_obj = jnp.load("results/vit_imagenette_psmile0_predsaver_sampling.npz", allow_pickle=True)
predsaver_obj_de = jnp.load("results/vit_imagenette_psmile0_predsaver_de.npz", allow_pickle=True)
# %%
target = predsaver_obj["target"]
target.shape
# %%
pred_dist_de = predsaver_obj_de["pred_dist"]
pred_dist_de.shape # (num_chains, 1, num_data, num_classes)
# %%
pred_dist = predsaver_obj["pred_dist"]
pred_dist.shape # (num_chains, num_samples, num_data, num_classes)

# %%
def get_pred_labels(pred_dist):
    pred_labels = jnp.argmax(pred_dist, axis=-1)
    return pred_labels
# %%
full_ensemble_acc = bmetrics.accuracy(pred=get_pred_labels(pred_dist), target=target)
full_chainwise_acc = bmetrics.accuracy(pred=get_pred_labels(pred_dist), target=target, chainwise=True)
print("Full ensemble accuracy:", full_ensemble_acc)
print("Chainwise accuracy:", full_chainwise_acc)

# %% now lppd
full_ensemble_lppd = bmetrics.lppd(
    bmetrics.lppd_pointwise(pred_dist=pred_dist, y=target, 
                            task=Task.CLASSIFICATION)
)
print("Full ensemble LPPD:", full_ensemble_lppd)
# now chainwise (loop over first axis no chainwise argument available)
chainwise_lppd = [
    bmetrics.lppd(
        bmetrics.lppd_pointwise(pred_dist=pred_dist[chain_idx:chain_idx+1], y=target, 
                                task=Task.CLASSIFICATION).squeeze(0)
    )
    for chain_idx in range(pred_dist.shape[0])
]
print("Chainwise LPPD:", [float(lppd) for lppd in chainwise_lppd])
# %%

# now implement the eValue logic!
