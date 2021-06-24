import streamlit as st
import torch
import torchvision.models as models
import deep_data_profiler as ddp
import matplotlib.pyplot as plt


if __name__ == "__main__":
    st.set_page_config(
        page_title="Spectral Analysis",
    )
    st.title("Spectral Analysis")

    ignore_set = {
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
    }

    model_options = [
        model
        for model in list(models.__dict__.keys())
        if model not in ignore_set
    ]
    model_str = st.sidebar.selectbox(
        "Choose PyTorch torchvision model architecture", options=model_options
    )

    uploaded_file = st.sidebar.file_uploader("PyTorch model weights", type=['pt','pth'])
    use_pretrained_weights = st.sidebar.radio(label="Use PyTorch pre-trained weights", options=["Pre-trained", "Random initialization"])

    if use_pretrained_weights == "Pre-trained":
        model = models.__dict__[model_str](pretrained=True).eval()
    else:
        model = models.__dict__[model_str](pretrained=False).eval()

    try:
        analysis = ddp.SpectralAnalysis(model)
    except Exception as e:
        st.write(e)

    # compute the SVD of X for each layer, return in dict
    eigenvalue_dict = analysis.spectral_analysis()

    # fit a power law distribution for spectral distribution
    # computed, per layer
    alpha_dict = analysis.fit_power_law(eig_dict=eigenvalue_dict)

    # threshold on the "fat-tailedness" of the power-law distribution
    layer_phenomenology = analysis.layer_RMT(alpha_dict=alpha_dict)

    # iterate through the final layers, which the dicts use as keys
    layers = list(alpha_dict.items())
    for idx, (layer, _) in enumerate(layers):

        # grab eigenvalues from the trained
        eigenvalues, _ = eigenvalue_dict[layer]

        # power law alphas
        alpha, _ = alpha_dict[layer]

        # and get the per-layer regularization predictions
        phenom = layer_phenomenology[idx]


        fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(2, 2))

        axs.hist(eigenvalues, bins="auto", density=True)
        axs.set_title("Network \n power-law fit" + fr" $\alpha$ = {round(alpha, 1)}", fontsize = 4)
        axs.set_xlabel('Eigenvalues of $X$', fontsize = 4)
        axs.set_ylabel('ESD', fontsize = 4)
        fig.suptitle(f"Layer {layer} spectral distribution", fontsize=6)
        st.pyplot(fig)
