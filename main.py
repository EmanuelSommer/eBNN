# %%
from bayesmates.config.data import Task
import bayesmates.inference.metrics as bmetrics
import plotnine as pn
import jax.numpy as jnp
import pandas as pd
import jax
from tqdm import tqdm

########################################################################################
# Load & prep data + baseline calculations
########################################################################################

# %% ViT on imagenette with psmile
predsaver_obj = jnp.load("results/vit_imagenette_psmile0_predsaver_sampling.npz", allow_pickle=True)
predsaver_obj_de = jnp.load("results/vit_imagenette_psmile0_predsaver_de.npz", allow_pickle=True)
exp_id = "vit_imagenette_psmile0"
# %% Resnet7 on cifar10 with psmile
predsaver_obj = jnp.load("results/resnet7_cifar10_psmile0_predsaver_sampling.npz", allow_pickle=True)
predsaver_obj_de = jnp.load("results/resnet7_cifar10_psmile0_predsaver_de.npz", allow_pickle=True)
exp_id = "resnet7_cifar10_psmile0"
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
# Split into validation and test sets (30% val, 70% test)
num_data = pred_dist.shape[2]
val_size = int(0.3 * num_data)
test_size = num_data - val_size
print("Validation size:", val_size)
print("Test size:", test_size)
val_indices = jnp.arange(0, val_size)
test_indices = jnp.arange(val_size, num_data)
target_val = target[val_indices]
target_test = target[test_indices]
pred_dist_val = pred_dist[:, :, val_indices, :]
pred_dist_test = pred_dist[:, :, test_indices, :]
pred_dist_de_val = pred_dist_de[:, :, val_indices, :]
pred_dist_de_test = pred_dist_de[:, :, test_indices, :]

# %%
def get_pred_labels(pred_dist):
    pred_labels = jnp.argmax(pred_dist, axis=-1)
    return pred_labels
# %%
full_ensemble_acc = bmetrics.accuracy(pred=get_pred_labels(pred_dist_test), target=target_test)
full_chainwise_acc = bmetrics.accuracy(pred=get_pred_labels(pred_dist_test), target=target_test, chainwise=True)
print("Full ensemble accuracy:", full_ensemble_acc)
print("Chainwise accuracy:", full_chainwise_acc)

# %% now lppd
full_ensemble_lppd = bmetrics.lppd(
    bmetrics.lppd_pointwise(pred_dist=pred_dist_test, y=target_test, 
                            task=Task.CLASSIFICATION)
)
print("Full ensemble LPPD:", full_ensemble_lppd)
# now chainwise (loop over first axis no chainwise argument available)
chainwise_lppd = [
    bmetrics.lppd(
        bmetrics.lppd_pointwise(pred_dist=pred_dist_test[chain_idx:chain_idx+1], y=target_test, 
                                task=Task.CLASSIFICATION).squeeze(0)
    )
    for chain_idx in range(pred_dist_test.shape[0])
]
print("Chainwise LPPD:", [float(lppd) for lppd in chainwise_lppd])
# %% chainwise lppd de
chainwise_lppd_de = [
    bmetrics.lppd(
        bmetrics.lppd_pointwise(pred_dist=pred_dist_de_test[chain_idx:chain_idx+1], y=target_test, 
                                task=Task.CLASSIFICATION).squeeze(0)
    )
    for chain_idx in range(pred_dist_de_test.shape[0])
]
print("Chainwise LPPD DE:", [float(lppd) for lppd in chainwise_lppd_de])
# %%
chainwise_acc_de = bmetrics.accuracy(
    pred=get_pred_labels(pred_dist_de_test),
    target=target_test,
    chainwise=True
)
print("Chainwise accuracy DE:", chainwise_acc_de)


########################################################################################
# E-values calculation
########################################################################################
# %%
def get_logprob_up_to_k(pred_dist, target, k):
    # note everything is done per chain
    # use all samples up to k
    pred_dist_k = pred_dist[:, :k, :, :]  # (num_chains, k, num_data, num_classes)
    # compute pointwise the logprob
    logprob_pw = bmetrics.lppd_pointwise(pred_dist=pred_dist_k, y=target, task=Task.CLASSIFICATION)
    # logmeanexp over the samples
    logprob_pw_lme = jax.nn.logsumexp(logprob_pw, axis=1, b = 1 / logprob_pw.shape[1])  # (num_chains, num_data)
    # sum over data points
    logprob_up_to_k = jnp.sum(logprob_pw_lme, axis=-1)  # (num_chains,)
    return logprob_up_to_k

@jax.jit
def get_evalue_from_logprops(logprob_up_to_k, logprob_reference):
    e_values = jnp.exp(logprob_up_to_k - logprob_reference)
    return e_values

# %%
# logprob_reference = get_logprob_up_to_k(pred_dist=pred_dist_de_val, target=target_val, k=pred_dist_de_val.shape[1]) # de reference
logprob_reference = get_logprob_up_to_k(pred_dist=pred_dist_val, target=target_val, k=1) # first sample reference

# now build up the e-values chainwise for increasing k
max_k = pred_dist_val.shape[1]
evalues_chainwise = []
for k in tqdm(range(1, max_k + 1)):
    logprob_up_to_k = get_logprob_up_to_k(pred_dist=pred_dist_val, target=target_val, k=k)  # (num_chains,)
    e_values = get_evalue_from_logprops(logprob_up_to_k, logprob_reference)  # (num_chains,)
    evalues_chainwise.append(e_values)
evalues_chainwise = jnp.stack(evalues_chainwise, axis=1)  # (num_chains, max_k)
print("E-values chainwise shape:", evalues_chainwise.shape)

########################################################################################
# E-value visualization and early stopping analysis
########################################################################################

# %%
# Vizualize e-values: x axis samples, group by chain, evolvement as lines (evalues on y axis) + horizontal lines at 10, 20 and 100
evalues_chainwise_df = pd.DataFrame(evalues_chainwise.reshape(-1, evalues_chainwise.shape[-1]).T)
evalues_chainwise_df = pd.melt(evalues_chainwise_df.reset_index(), id_vars=["index"], var_name="chain", value_name="e_value")
evalues_chainwise_df = evalues_chainwise_df.rename(columns={"index": "num_samples"})
# make chain a float
evalues_chainwise_df["chain"] = evalues_chainwise_df["chain"].astype(float)
evalues_chainwise_df
# %%
plot = (pn.ggplot(evalues_chainwise_df) +
 pn.aes(x="num_samples", y="e_value", group="chain") +
 pn.geom_line(alpha=0.7, color="#324b94") +
 pn.geom_hline(yintercept=100, linetype="dashed", color="#4c915c", size=1) +
 pn.scale_y_log10(breaks=[0, 1, 100], labels=["0", "1", "100 ($\\alpha = 0.01$)"]) +
 pn.labs(title="", y="E-value (log scale)", x="Number of samples") +
 pn.theme_minimal() + 
 pn.theme(
    figure_size=(6,4),
    text=pn.element_text(size=11),
    axis_text=pn.element_text(size=10)
 )
)

# save the plot as pdf
plot.save(f"results/plots/{exp_id}_evalues_chainwise.pdf", dpi=300)
plot

# %% now calculate the early stopping sample sizes per chain for alphas in [0.01, 0.05, 0.1]
alphas = [0.001,0.01, 0.05, 0.1]
early_stopping_samples = {alpha: [] for alpha in alphas}
for chain_idx in range(evalues_chainwise.shape[0]):
    e_values_chain = evalues_chainwise[chain_idx, :]  # (max_k,)
    for alpha in alphas:
        threshold = 1 / alpha
        # find first index where e_value exceeds threshold
        exceed_indices = jnp.where(e_values_chain >= threshold)[0]
        if exceed_indices.shape[0] > 0:
            early_stopping_sample = exceed_indices[0] + 1  # +1 because indices start at 0
        else:
            early_stopping_sample = max_k  # did not exceed threshold
        early_stopping_samples[alpha].append(early_stopping_sample)
# %% add in a dataframe the average early stopping samples per alpha and std
early_stopping_data = []
for alpha in alphas:
    mean_early_stopping = jnp.mean(jnp.array(early_stopping_samples[alpha]))
    std_early_stopping = jnp.std(jnp.array(early_stopping_samples[alpha]))
    early_stopping_data.append({
        "alpha": alpha,
        "mean_early_stopping_samples": mean_early_stopping,
        "std_early_stopping_samples": std_early_stopping
    })
early_stopping_df = pd.DataFrame(early_stopping_data)
early_stopping_df
# %%
# visualize the early stopping sample sizes as distributions per alpha
early_stopping_long_df = pd.melt(
    pd.DataFrame(early_stopping_samples),
    var_name="alpha",
    value_name="early_stopping_samples"
)
early_stopping_long_df["alpha"] = early_stopping_long_df["alpha"].astype(float)
early_stopping_long_df["early_stopping_samples"] = early_stopping_long_df["early_stopping_samples"].astype(int)
plot = (pn.ggplot(early_stopping_long_df) +
 pn.aes(x="early_stopping_samples", fill="factor(alpha)") +
 pn.geom_histogram(bins=25, position="dodge", alpha=0.5) +
#  pn.geom_density(alpha=0.7) +
#  pn.geom_rug(alpha=0.5) +
 pn.labs(title="", x="Early Stopping sample size", y="Count", fill="$\\alpha$") +
 pn.theme_minimal() +
 pn.theme(
    figure_size=(6,4),
    text=pn.element_text(size=11),
    axis_text=pn.element_text(size=10)
 )
)

# save the plot as pdf
plot.save(f"results/plots/{exp_id}_early_stopping_sample_sizes.pdf", dpi=300)
plot    

# %% now based on the early stopping sample sizes select samples only that have an evalue exceeding the threshold and compute test accuracy and lppd
# skip if early stopping sample size is max_k (no early stopping) then report None
results_early_stopping = []
for alpha in alphas:
    threshold = 1 / alpha
    for chain_idx in range(evalues_chainwise.shape[0]):
        early_stopping_sample = early_stopping_samples[alpha][chain_idx]
        if early_stopping_sample == max_k:
            # no early stopping
            test_acc = None
            test_lppd = None
            early_stopping_sample = 0
        else:
            pred_dist_es = pred_dist_test[chain_idx:chain_idx+1, :early_stopping_sample, :, :]  # (1, k_es, num_data, num_classes)
            # compute test accuracy
            test_acc = bmetrics.accuracy(
                pred=get_pred_labels(pred_dist_es),
                target=target_test,
                chainwise=False
            )
            # compute test lppd
            test_lppd = bmetrics.lppd(
                bmetrics.lppd_pointwise(pred_dist=pred_dist_es, y=target_test, task=Task.CLASSIFICATION)
            )
        results_early_stopping.append({
            "alpha": alpha,
            "chain": chain_idx,
            "early_stopping_sample": early_stopping_sample,
            "test_accuracy": test_acc,
            "test_lppd": test_lppd
        })

# %%
# for reference now use the chainwise results with all samples
# aggregate the chainwise results both accuracy and lppd for the early stopping (per alpha) and full samples
for chain_idx in range(evalues_chainwise.shape[0]):
    pred_dist_full = pred_dist_test[chain_idx:chain_idx+1, :, :, :]  # (1, num_samples, num_data, num_classes)
    test_acc_full = bmetrics.accuracy(
        pred=get_pred_labels(pred_dist_full),
        target=target_test,
        chainwise=False
    )
    test_lppd_full = bmetrics.lppd(
        bmetrics.lppd_pointwise(pred_dist=pred_dist_full, y=target_test, task=Task.CLASSIFICATION)
    )
    results_early_stopping.append({
        "alpha": "full",
        "chain": chain_idx,
        "early_stopping_sample": max_k,
        "test_accuracy": test_acc_full,
        "test_lppd": test_lppd_full
    })

results_early_stopping_df = pd.DataFrame(results_early_stopping)
results_early_stopping_df
# %%
########################################################################################
# E-value based early stopping performance summary and visualization
########################################################################################

# group by alpha and compute mean and std of test accuracy and lppd
summary_early_stopping_df = results_early_stopping_df.groupby("alpha").agg(
    mean_test_accuracy=("test_accuracy", "mean"),
    std_test_accuracy=("test_accuracy", "std"),
    mean_test_lppd=("test_lppd", "mean"),
    std_test_lppd=("test_lppd", "std")
).reset_index()
summary_early_stopping_df
# %% append the average number of samples used for early stopping per alpha
mean_early_stopping_samples = results_early_stopping_df.groupby("alpha")["early_stopping_sample"].mean().reset_index()
mean_early_stopping_samples = mean_early_stopping_samples.rename(columns={"early_stopping_sample": "mean_early_stopping_samples"})
summary_early_stopping_df = summary_early_stopping_df.merge(mean_early_stopping_samples, on="alpha", how="left")
summary_early_stopping_df

# %% visualize the summary results with facet wrap for accuracy and lppd
summary_early_stopping_long_df = pd.melt(
    summary_early_stopping_df,
    id_vars=["alpha", "mean_early_stopping_samples"],
    value_vars=["mean_test_accuracy", "mean_test_lppd"],
    var_name="metric",
    value_name="mean_value"
)

# Add corresponding std columns
std_mapping = {"mean_test_accuracy": "std_test_accuracy", "mean_test_lppd": "std_test_lppd"}
summary_early_stopping_long_df["std_value"] = summary_early_stopping_long_df.apply(
    lambda row: summary_early_stopping_df.loc[
        summary_early_stopping_df["alpha"] == row["alpha"], 
        std_mapping[row["metric"]]
    ].iloc[0], axis=1
)

# Create cleaner metric labels
summary_early_stopping_long_df["metric_label"] = summary_early_stopping_long_df["metric"].map({
    "mean_test_accuracy": "Test Accuracy",
    "mean_test_lppd": "Test LPPD"
})

summary_early_stopping_long_df["mean_value"] = pd.to_numeric(summary_early_stopping_long_df["mean_value"], errors="coerce")
summary_early_stopping_long_df["std_value"] = pd.to_numeric(summary_early_stopping_long_df["std_value"], errors="coerce")
summary_early_stopping_long_df["mean_early_stopping_samples"] = pd.to_numeric(summary_early_stopping_long_df["mean_early_stopping_samples"], errors="coerce")
summary_early_stopping_long_df["ymin"] = summary_early_stopping_long_df["mean_value"] - summary_early_stopping_long_df["std_value"]
summary_early_stopping_long_df["ymax"] = summary_early_stopping_long_df["mean_value"] + summary_early_stopping_long_df["std_value"]
summary_early_stopping_long_df["alpha"] = summary_early_stopping_long_df["alpha"].astype(str)

de_lppd_mean = jnp.mean(jnp.array(chainwise_lppd_de)).item()
de_lppd_std = jnp.std(jnp.array(chainwise_lppd_de)).item()
de_acc_mean = jnp.mean(jnp.array(chainwise_acc_de)).item()
de_acc_std = jnp.std(jnp.array(chainwise_acc_de)).item()
de_lines_df = pd.DataFrame({
    "metric_label": ["Test LPPD", "Test Accuracy"],
    "de_mean": [de_lppd_mean, de_acc_mean],
    "de_ymin": [de_lppd_mean - de_lppd_std, de_acc_mean - de_acc_std],
    "de_ymax": [de_lppd_mean + de_lppd_std, de_acc_mean + de_acc_std],
})
alpha_levels = list(summary_early_stopping_long_df["alpha"].unique())
de_band_df = (
    de_lines_df.assign(_k=1)
    .merge(pd.DataFrame({"alpha": alpha_levels, "_k": [1] * len(alpha_levels)}), on="_k")
    .drop(columns="_k")
)

plot = (pn.ggplot(summary_early_stopping_long_df) +
 pn.aes(x="alpha", y="mean_value") +
 pn.geom_hline(data=de_lines_df, mapping=pn.aes(yintercept="de_mean"),
               linetype="dashed", color="#ff7f0e", size=1, inherit_aes=False) +
 pn.geom_ribbon(data=de_band_df, mapping=pn.aes(x="alpha", ymin="de_ymin", ymax="de_ymax", group="metric_label"),
                fill="#ff7f0e", alpha=0.2, inherit_aes=False) +
 pn.geom_errorbar(pn.aes(ymin="ymin", ymax="ymax", color="mean_early_stopping_samples"),
                  width=0.001, size=1, alpha=0.8) +
 pn.geom_point(pn.aes(color="mean_early_stopping_samples"), alpha=1, size=2, shape="D") +
 pn.facet_wrap("~ metric_label", scales="free_y", ncol=2) +
 pn.labs(title="", x="Early stopping level ($\\alpha$)", y="Value ($\\pm$ SD)", color="Mean early\nstopping samples") +
 pn.theme_minimal() +
 pn.theme(legend_position="bottom", figure_size=(8,4),
    text=pn.element_text(size=11),
    axis_text=pn.element_text(size=10)) +
 pn.scale_color_gradient(low="#e04f93", high="#032459")
)

# save the plot as pdf
plot.save(f"results/plots/{exp_id}_early_stopping_summary.pdf", dpi=300)
plot

# %%
########################################################################################
# Influence on ensemble metrics
########################################################################################

no_early_stop_strategy = "none" # "none" or "first"

ensemble_results_early_stopping = []
for alpha in tqdm(alphas):
    pred_dists_es = []
    for chain_idx in range(evalues_chainwise.shape[0]):
        early_stopping_sample = early_stopping_samples[alpha][chain_idx]
        if early_stopping_sample != max_k:
            pred_dist_es = pred_dist_test[chain_idx:chain_idx+1, :early_stopping_sample, :, :]  # (1, k_es, num_data, num_classes)
            pred_dists_es.append(pred_dist_es)
        else:
            if no_early_stop_strategy == "none":
                # skip this chain
                continue
            elif no_early_stop_strategy == "first":
                # use first sample only
                pred_dist_es = pred_dist_test[chain_idx:chain_idx+1, :1, :, :]  # (1, 1, num_data, num_classes)
                pred_dists_es.append(pred_dist_es)
            else:
                raise ValueError("Invalid no_early_stop_strategy")
    pred_dist_es_ensemble = jnp.concatenate(pred_dists_es, axis=1)  # (num_chains, k_es_chain, num_data, num_classes)
    ensemble_acc_es = bmetrics.accuracy(
        pred=get_pred_labels(pred_dist_es_ensemble),
        target=target_test,
        chainwise=False
    )
    ensemble_lppd_es = bmetrics.lppd(
        bmetrics.lppd_pointwise(pred_dist=pred_dist_es_ensemble, y=target_test, task=Task.CLASSIFICATION)
    )
    ensemble_results_early_stopping.append({
        "alpha": alpha,
        "ensemble_accuracy": ensemble_acc_es,
        "ensemble_lppd": ensemble_lppd_es,
        "ensemble_members": pred_dist_es_ensemble.shape[1]
    })
ensemble_results_early_stopping.append({
    "alpha": "full",
    "ensemble_accuracy": full_ensemble_acc,
    "ensemble_lppd": full_ensemble_lppd,
    "ensemble_members": pred_dist_test.shape[0] * pred_dist_test.shape[1]
})
ensemble_results_early_stopping.append({
    "alpha": "DE",
    "ensemble_accuracy": bmetrics.accuracy(
        pred=get_pred_labels(pred_dist_de_test),
        target=target_test,
        chainwise=False
    ),
    "ensemble_lppd": bmetrics.lppd(
        bmetrics.lppd_pointwise(pred_dist=pred_dist_de_test, y=target_test, task=Task.CLASSIFICATION)
    ),
    "ensemble_members": pred_dist_de_test.shape[0] * pred_dist_de_test.shape[1]
})
ensemble_results_early_stopping_df = pd.DataFrame(ensemble_results_early_stopping)
ensemble_results_early_stopping_df
# %%
# Visualize ensemble results
# Coerce jnp arrays to floats before plotting
ensemble_results_early_stopping_df = ensemble_results_early_stopping_df.copy()
for col in ["ensemble_accuracy", "ensemble_lppd", "ensemble_members"]:
    ensemble_results_early_stopping_df[col] = ensemble_results_early_stopping_df[col].apply(lambda x: float(x))

ensemble_results_long_df = pd.melt(
    ensemble_results_early_stopping_df,
    id_vars=["alpha", "ensemble_members"],
    value_vars=["ensemble_accuracy", "ensemble_lppd"],
    var_name="metric",
    value_name="value"
)
# Create cleaner metric labels
ensemble_results_long_df["metric_label"] = ensemble_results_long_df["metric"].map({
    "ensemble_accuracy": "Ensemble Accuracy",
    "ensemble_lppd": "Ensemble LPPD"
})
# Ensure consistent types for plotting
ensemble_results_long_df["alpha"] = ensemble_results_long_df["alpha"].astype(str)
# order alpha such that DE is first then the alphas in increasing order then full
alpha_order = ["DE"] + [str(alpha) for alpha in sorted(alphas)] + ["full"]
ensemble_results_long_df["alpha"] = pd.Categorical(ensemble_results_long_df["alpha"], categories=alpha_order, ordered=True)
ensemble_results_long_df["ensemble_members"] = pd.to_numeric(ensemble_results_long_df["ensemble_members"], errors="coerce")
ensemble_results_long_df["value"] = pd.to_numeric(ensemble_results_long_df["value"], errors="coerce")

# use a scatterplot with lines connecting the points
plot = (pn.ggplot(ensemble_results_long_df) +
 pn.aes(x="alpha", y="value") +
 pn.geom_point(pn.aes(size="ensemble_members"), alpha=0.8, color="#324b94") +
 pn.geom_line(group=1, linetype="dashed", color="#324b94", alpha=0.5) +
 pn.facet_wrap("~ metric_label", scales="free_y", ncol=2) +
 pn.labs(title="", x="Early stopping level ($\\alpha$)", y="Value", size="Ensemble members") +
 pn.theme_minimal() +
 pn.theme(
    legend_position="bottom", figure_size=(7,3),
    text=pn.element_text(size=11),
    axis_text=pn.element_text(size=10)
)
)

# save the plot as pdf
plot.save(f"results/plots/{exp_id}_ensemble_results_early_stopping.pdf", dpi=300)
plot
# %%
