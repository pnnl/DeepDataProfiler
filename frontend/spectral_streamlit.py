import streamlit as st
import torch
import torchvision.models as models
import deep_data_profiler as ddp
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import pandas as pd
import os
from spectral_helpers import get_LeNet


def show_svd():
    st.title("Spectral Analysis")

    about = """
    #### Overview
    Spectral Analysis is based on methods originating from Random Matrix theory,
    brought to deep neural networks by Martin and Mahoney.

    The core observation these metrics use is that the covariance of the weight matrices $X = W^TW$,
    of DNN layers start out looking like the right plot below. As these layers are trained by Stochastic Gradient Descent,
    the ESD (empirical spectral distribution) of $X$ **systematically gain fat-tails**, like the plot on the left
    (which is a layer that has been trained on ImageNet).

    We can use this measure to know how much a layer (or entire model) has been trained, and how likely it is that a layer (or model) has been overfit.
    """
    """
    The major improvement we make over the above work is our handling of
    convolutional layers: our methods are more predictive, more principled, and over an
    order of magnitude faster than the code released by the authors in
    https://github.com/CalculatedContent/WeightWatcher.

    #### References
    For example, see:

    - Martin, Charles H., Tongsu Serena Peng, and Michael W. Mahoney. "Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data." Nature Communications 12.1 (2021): 1-13.

    - Mahoney, Michael, and Charles Martin. "Traditional and heavy tailed self regularization in neural network models." International Conference on Machine Learning. PMLR, 2019.
    """

    with st.beta_expander(
        "Background for this tool",
    ):
        st.write(about)
        path = os.path.dirname(__file__)
        st.image(os.path.join(path, "data/new_spectral_plot.png"), width=None)

    ignore_set = {
        "alexnet",
        "AlexNet",
        "DenseNet",
        "GoogLeNet",
        "GoogLeNetOutputs",
        "Inception3",
        "InceptionOutputs",
        "MNASNet",
        "MobileNetV2",
        "MobileNetV3",
        "ResNet",
        "ShuffleNetV2",
        "SqueezeNet",
        "VGG",
        "_GoogLeNetOutputs",
        "_InceptionOutputs",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "_utils",
        "densenet",
        "detection",
        "inception",
        "mnasnet",
        "mnasnet0_75",
        "mnasnet1_3",
        "mobilenet",
        "mobilenetv2",
        "mobilenetv3",
        "quantization",
        "resnet",
        "segmentation",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
        "shufflenetv2",
        "squeezenet",
        "utils",
        "vgg",
        "video",
        "googlenet",  # trained vs. pre-trained have different layer numbers???
    }

    model_list = list(models.__dict__.keys())
    model_list.append("LeNet")

    model_options = [model for model in model_list if model not in ignore_set]
    with st.beta_expander("Select model", expanded=True):
        with st.form(key="my_form"):
            model_str = st.selectbox(
                "Choose a PyTorch torchvision model architecture",
                options=model_options,
            )

            uploaded_dict = st.file_uploader(
                "PyTorch model weights (defaults to ImageNet-1k training if not provided)",
                type=["pt", "pth"],
            )

            if uploaded_dict:
                try:
                    loaded_dict = torch.load(uploaded_dict)
                    try:
                        if "model_state_dict" in loaded_dict:
                            st.write("using state dict")
                            loaded_dict = loaded_dict["model_state_dict"]
                        elif "state_dict" in loaded_dict:
                            st.write("using state dict")
                            loaded_dict = loaded_dict["state_dict"]
                    except:
                        pass
                except:
                    loaded_dict = 0
                    st.write("Cannot except this file")
                if loaded_dict:
                    model = models.__dict__[model_str](pretrained=True)
                    model.load_state_dict(loaded_dict)
                    model.eval()
                    random_model = models.__dict__[model_str](pretrained=False).eval()
                    # try:
                    #     model = models.__dict__[model_str](pretrained=False).load_state_dict(loaded_dict)
                    #     model.eval()
                    # except:
                    #     st.write("For now, you need to provide the correct architecture")

            elif model_str == "LeNet":
                model = get_LeNet()
                random_model = get_LeNet()
            else:
                model = models.__dict__[model_str](pretrained=True).eval()
                random_model = models.__dict__[model_str](
                    pretrained=False
                ).eval()

            try:
                with st.spinner("Computing spectral statistics..."):
                    analysis = ddp.SpectralAnalysis(model)
                    analysis_random = ddp.SpectralAnalysis(random_model)

                    # compute the SVD of X for each layer, return in dict
                    eigenvalue_dict = analysis.spectral_analysis()
                    eigenvalue_dict_random = (
                        analysis_random.spectral_analysis()
                    )

                    # fit a power law distribution for spectral distribution
                    # computed, per layer
                    alpha_dict = analysis.fit_power_law(
                        eig_dict=eigenvalue_dict
                    )
                    alpha_dict_random = analysis_random.fit_power_law(
                        eig_dict=eigenvalue_dict_random
                    )
                st.form_submit_button(label="Re-compute and plot spectral statistics")
            except Exception as e:
                st.write(e)

    with st.beta_expander("'Universal' capacity metric"):
        st.write(
            r"Returns the capacity metric defined by $\widehat{\alpha}=\frac{1}{L} \sum_{l} \alpha_{l} \log \lambda_{\max , l}$"
        )
        universal = analysis.universal_metric(alpha_dict=alpha_dict)
        universal_random = analysis_random.universal_metric(
            alpha_dict=alpha_dict_random
        )
        st.write(
            "**Univeral capacity metric of the uploaded model**: ", universal
        )
        st.write(
            "**Univeral capacity metric of a random model**: ",
            universal_random,
        )
    with st.beta_expander("Get metrics per-layer"):
        st.write(
            r"Metrics on the covariance metrics of the weights for each layer, i.e. $X = W W^T$. Fits with a powerlaw distribution $\rho(\lambda) \sim \lambda^{-\alpha}$ using the MLE from https://arxiv.org/abs/0706.1062."
        )
        # threshold on the "fat-tailedness" of the power-law distribution
        # iterate through the final layers, which the dicts use as keys
        layers = list(alpha_dict.items())
        layers_random = list(alpha_dict_random.items())

        alphas = np.array([layer[1][0] for layer in layers])
        alphas_random = np.array([layer[1][0] for layer in layers_random])

        df = pd.DataFrame(
            {
                "Per layer alpha metrics trained": alphas,
                f"Layer of {model_str}": np.array([layer for (layer, _) in layers]),
                f"Uploaded model": len(alphas) * ["Uploaded model"],
            }
        )
        df_random = pd.DataFrame(
            {
                "Per layer alpha metrics random": alphas_random,
                f"Layer of {model_str}": np.array([layer for (layer, _) in layers]),
                f"Random model": len(alphas) * ["Random model"],
            }
        )
        chart = (
            alt.Chart(df, title=f"Per layer alpha metrics of {model_str}")
            .mark_point()
            .encode(
                x=alt.X(f"Layer of {model_str}", title="Layer of model"),
                y=alt.Y(
                    "Per layer alpha metrics trained",
                    title="Per layer alpha metric",
                ),
                color=alt.value("red"),
                opacity="Uploaded model",
                tooltip=alt.Tooltip("Per layer alpha metrics trained", format=",.2f"),
            )
            .interactive()
        )

        chart_random = (
            alt.Chart(df_random, title=f"Per layer alpha metrics of {model_str}")
            .mark_point()
            .encode(
                x=alt.X(f"Layer of {model_str}", title="Layer of model"),
                y=alt.Y(
                    "Per layer alpha metrics random",
                    title="Per layer alpha metric",
                ),
                color=alt.value("blue"),
                shape="Random model",
                tooltip=alt.Tooltip("Per layer alpha metrics random", format=",.2f"),
            )
            .interactive()
        )

        st.altair_chart(chart + chart_random, use_container_width=True)

        per_layer = st.button("Plot distributions per layer")
        if per_layer:
            for idx, (layer, _) in enumerate(layers):

                fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 4))
                # grab eigenvalues from the trained
                eigenvalues, _ = eigenvalue_dict[layer]
                # power law alphas
                alpha, _ = alpha_dict[layer]
                axs[0].hist(
                    eigenvalues, bins="auto", density=True, color="red"
                )
                axs[0].set_title(
                    "Uploaded Network \n power-law fit"
                    + rf" $\alpha$ = {round(alpha, 1)}",
                    fontsize=14,
                )
                axs[0].set_xlabel("Eigenvalues of $X$", fontsize=14)
                axs[0].set_ylabel("ESD", fontsize=14)

                # grab eigenvalues from the random
                eigenvalues_random, _ = eigenvalue_dict_random[layer]
                # power law alphas
                alpha_random, _ = alpha_dict_random[layer]
                axs[1].hist(eigenvalues_random, bins="auto", density=True)
                axs[1].set_title(
                    "Random Network \n power-law fit"
                    + rf" $\alpha$ = {round(alpha_random, 1)}",
                    fontsize=14,
                )
                axs[1].set_xlabel("Eigenvalues of $X$", fontsize=14)
                axs[1].set_ylabel("ESD", fontsize=14)

                fig.suptitle(f"Layer {layer} spectral distribution", fontsize=16)
                st.pyplot(fig)

    # with st.beta_expander("Plot metric during training"):
    #     if model_str == "LeNet":

    #         training_error = [0.56, 0.5, 0.46, 0.43, 0.41, 0.39, 0.37, 0.36, 0.34, 0.32]
    #         test_error = [0.53, 0.48, 0.46, 0.43, 0.42, 0.4, 0.39, 0.39, 0.38, 0.38]
    #         folder_path = st.text_input("Training folder path", value="../tutorials/spectral_data/cifar_files/")

    #         universal_metrics = []
    #         layer_to_observe = 3

    #         fig, axs = plt.subplots(4, 3, constrained_layout=True, figsize=(15, 15))

    #         for i in range(10):
    #             PATH = os.path.join(folder_path, f'cifar_net_{i}.pth')
    #             model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu') ))

    #             # define SpectralAnalysis, and calculate the model metric
    #             analysis = ddp.SpectralAnalysis(model)
    #             # compute the SVD of X for each layer, return in dict
    #             eigenvalue_dict = analysis.spectral_analysis()
    #             # fit a power law distribution for each spectral distribution computed, per layer
    #             alpha_dict = analysis.fit_power_law(eig_dict=eigenvalue_dict)

    #             # choose an arbitrary layer to plot
    #             eigenvalues, _ = eigenvalue_dict[layer_to_observe]
    #             alpha, _ = alpha_dict[layer_to_observe]
    #             axs[i//3, i%3].hist(eigenvalues, bins="auto", density=True)
    #             axs[i//3, i%3].set_title(
    #                 (fr"Epoch {i}, power-law fit $\alpha = {round(alpha, 2)}$"
    #                  + f"\n train error: {int(training_error[i]*100)}% | test error: {int(test_error[i]*100)}%"
    #                                      ), fontsize=18)
    #             axs[i//3, i%3].set_xlabel('Eigenvalues of $X$', fontsize = 14)
    #             axs[i//3, i%3].set_ylabel('ESD', fontsize = 14)

    #             # collect the universal alpha metrics, per epoch
    #             universal_metrics.append(analysis.universal_metric(alpha_dict=alpha_dict))

    #         fig.suptitle(f"Training epoch spectral distributions, for layer {layer_to_observe}", fontsize=16)
    #         axs[-1, -1].axis('off')
    #         axs[-1, -2].axis('off')
    #         st.pyplot(fig)

    #         # second plot with x-axis per epoch
    #         fig, ax1 = plt.subplots(figsize=(4, 4))
    #         ax1.set_xlabel('Epoch')
    #         ax1.set_ylabel('Percent error', color='tab:red')
    #         ax1.plot(training_error, '-o', color='tab:red', label="training error")
    #         ax1.plot(test_error,'-o', color='tab:orange', label="test error")

    #         ax1.tick_params(axis='y', labelcolor='tab:red')
    #         plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    #         ax2 = ax1.twinx()

    #         color = 'darkblue'
    #         ax2.set_ylabel('alpha', color=color)  # we already handled the x-label with ax1
    #         ax2.plot(universal_metrics, '-o', color=color, label="alpha metric")
    #         ax2.tick_params(axis='y', labelcolor=color)

    #         plt.legend(bbox_to_anchor=(1.15, 0.85), loc='upper left')

    #         # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #         plt.title("LeNet metric and loss vs training epoch")
    #         st.pyplot(fig)


#         else:
#             st.write("Per-epoch feature only implemented for LeNet")

if __name__ == "__main__":
    show_svd()
