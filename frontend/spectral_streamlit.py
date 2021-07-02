import streamlit as st
import torch
import torchvision.models as models
import deep_data_profiler as ddp
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import pandas as pd


if __name__ == "__main__":
    st.set_page_config(
        page_title="Spectral Analysis",
        page_icon=":chart_with_upwards_trend:",
    )
    st.title("Spectral Analysis")

    about = """
    Spectral Analysis is based on methods originating from Random Matrix theory,
    brought to deep neural networks by Martin and Mahoney. For example, see: [Traditional and Heavy-Tailed Self Regularization in Neural Network Models](https://arxiv.org/abs/1901.08276/) by Martin and Mahoney

    These methods act only on the weights of the Fully Connected and
    Convolutional layers a deep neural network. Despite this, they have
    proven effective in predicting

    1. Test accuracies with no access to the data distribution on which it was trained OR tested

    2. Relative performance between models of similar architecture classes

    3. Model and architecture improvements while training

    The major improvement we make over the above work is our handling of
    convolutional layers: our methods are more principled, and over an
    order of magnitude faster than the code released by the authors in
    https://github.com/CalculatedContent/WeightWatcher.
    """

    with st.beta_expander("Background for this tool"):
        st.write(about)

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

    model_options = [
        model
        for model in list(models.__dict__.keys())
        if model not in ignore_set
    ]
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
                random_model = models.__dict__[model_str](
                    pretrained=False
                ).eval()
                # try:
                #     model = models.__dict__[model_str](pretrained=False).load_state_dict(loaded_dict)
                #     model.eval()
                # except:
                #     st.write("For now, you need to provide the correct architecture")

        else:
            model = models.__dict__[model_str](pretrained=True).eval()
            random_model = models.__dict__[model_str](pretrained=False).eval()

        try:
            with st.spinner("Computing spectral statistics..."):
                analysis = ddp.SpectralAnalysis(model)
                analysis_random = ddp.SpectralAnalysis(random_model)

                # compute the SVD of X for each layer, return in dict
                eigenvalue_dict = analysis.spectral_analysis()
                eigenvalue_dict_random = analysis_random.spectral_analysis()

                # fit a power law distribution for spectral distribution
                # computed, per layer
                alpha_dict = analysis.fit_power_law(eig_dict=eigenvalue_dict)
                alpha_dict_random = analysis_random.fit_power_law(
                    eig_dict=eigenvalue_dict_random
                )
            st.form_submit_button(
                label="Re-compute and plot spectral statistics"
            )
        except Exception as e:
            st.write(e)

    # threshold on the "fat-tailedness" of the power-law distribution
    # iterate through the final layers, which the dicts use as keys
    layers = list(alpha_dict.items())
    layers_random = list(alpha_dict_random.items())

    alphas = np.array([layer[1][0] for layer in layers])
    alphas_random = np.array([layer[1][0] for layer in layers_random])

    df = pd.DataFrame(
        {
            "Per layer alpha metrics trained": alphas,
            f"Layer of {model_str}": np.array(
                [layer for (layer, _) in layers]
            ),
            f"Uploaded model": len(alphas) * ["Uploaded model"],
        }
    )
    df_random = pd.DataFrame(
        {
            "Per layer alpha metrics random": alphas_random,
            f"Layer of {model_str}": np.array(
                [layer for (layer, _) in layers]
            ),
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
            tooltip=alt.Tooltip(
                "Per layer alpha metrics trained", format=",.2f"
            ),
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
            tooltip=alt.Tooltip(
                "Per layer alpha metrics random", format=",.2f"
            ),
        )
        .interactive()
    )

    st.altair_chart(chart + chart_random, use_container_width=True)

    per_layer = st.button("Plot distributions per layer")
    if per_layer:
        for idx, (layer, _) in enumerate(layers):

            fig, axs = plt.subplots(
                1, 2, constrained_layout=True, figsize=(15, 4)
            )
            # grab eigenvalues from the trained
            eigenvalues, _ = eigenvalue_dict[layer]
            # power law alphas
            alpha, _ = alpha_dict[layer]
            axs[0].hist(eigenvalues, bins="auto", density=True)
            axs[0].set_title(
                "Uploaded Network \n power-law fit"
                + fr" $\alpha$ = {round(alpha, 1)}",
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
                + fr" $\alpha$ = {round(alpha_random, 1)}",
                fontsize=14,
            )
            axs[1].set_xlabel("Eigenvalues of $X$", fontsize=14)
            axs[1].set_ylabel("ESD", fontsize=14)

            fig.suptitle(f"Layer {layer} spectral distribution", fontsize=16)
            st.pyplot(fig)
