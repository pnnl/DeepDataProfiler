import streamlit as st
import boto3

from PIL import Image
from io import BytesIO
from botocore.exceptions import ClientError

import pickle as pkl
from pathlib import Path
from itertools import zip_longest
import sys
import torchvision.models as tvmodels
import torch

import numpy as np
from collections import OrderedDict

from lucent_svd.lucent.optvis import render, objectives

from persim import plot_diagrams
import os

from streamlit_tda import load_class_labels_dicts
from streamlit_svd import TorchProfilerSpectral


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_paths():
    """Loads the dict keyed by features, with values of dicts of
    singular values and the relative image paths that maximize these singular values."""
    img_paths = pkl.load(open("data/light_paths.pkl", "rb"))

    return img_paths


@st.cache(
    show_spinner=False,
    hash_funcs={
        "builtins.PyCapsule": lambda _: None,
    },
    allow_output_mutation=True,
)
def load_model():
    """Loads a model"""
    model_str = "vgg16"

    model_pre = tvmodels.__dict__[model_str](
        pretrained=True
    ).eval()  # .to(device)
    return model_pre


@st.cache(
    show_spinner=False,
    allow_output_mutation=True,
)
def load_svd_dicts(model_pre):
    """Loads a dictionary of the SVDs per layer;
    Keyed by layer"""
    profile_pre = TorchProfilerSpectral(model_pre)
    svd_dict = profile_pre.create_svd()
    svd_dict_visualization = {
        v[0][0].replace(".", "_"): v[1] for _, v in svd_dict.items()
    }
    return svd_dict_visualization


@st.cache(
    show_spinner=False,
    allow_output_mutation=True,
)
def svd_visualization(model, svd_dict, layer, svd_num):
    """Renders an SVD feature visualization"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = render.render_vis(
        model.to(device).eval(),
        objectives.svd(layer, svd_num, svd_dict),
        progress=False,
    )
    return output[0][0]


@st.cache(ttl=600)
def read_pickle_file(filename):
    s3 = boto3.resource("s3")
    my_pickle = pkl.loads(
        s3.Bucket("ddp-streamlit-data").Object(filename).get()["Body"].read()
    )
    return my_pickle


@st.cache(ttl=600)
def read_image_file(filename):
    s3 = boto3.resource("s3")
    my_image = Image.open(
        BytesIO(
            s3.Bucket("ddp-streamlit-data")
            .Object(filename)
            .get()["Body"]
            .read()
        )
    )
    return my_image


@st.cache(ttl=600)
def list_image_files(pathname):
    s3 = boto3.client("s3")
    all_objects = s3.list_objects(Bucket="ddp-streamlit-data", Prefix=pathname)
    return [dct["Key"] for dct in all_objects["Contents"]]


if __name__ == "__main__":
    st.set_page_config(
        page_title="SVD Feature Visualization", page_icon=":lower_left_crayon:"
    )
    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )
    query_params = st.experimental_get_query_params()
    tabs = ["About", "SVD Feature Visualizations", "TDA Visualizations"]
    if "tab" in query_params:
        active_tab = query_params["tab"][0]
    else:
        active_tab = "SVD Feature Visualizations"

    if active_tab not in tabs:
        st.experimental_set_query_params(tab="About")
        active_tab = "About"

    li_items = "".join(
        f"""
        <li class="nav-item">
            <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}">{t}</a>
        </li>
        """
        for t in tabs
    )
    tabs_html = f"""
        <ul class="nav nav-tabs">
        {li_items}
        </ul>
    """

    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # an html hack to hide the top bar
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # NOTE: in future, allow user to supply relative path
    if active_tab == "About":
        st.header("About this tool")
        header = """
        This is a visualization tool for the [DeepDataProfiler](https://pnnl.github.io/DeepDataProfiler/build/index.html) library. For now, it consists of two separate components:

        1. Visualizations that link what we are calling the "SVD neurons" in a VGG-16 network with dataset examples from ImageNet.

        2. The persistence diagrams for DDP pofiler graphs for ImageNet images with VGG-16.

        Use the tabs above to navigate between these visualizations, or read more below.
        """
        st.write(header)
        body_svd = """
        ## 1 SVD Feature Visualizations
        ### Feature Visualization
        Feature visualization is an interpretability technique that optimizes an image so that it highly activates a neuron in a deep neural network (DNN). Feature visualizations have been used to gain a better understanding of how individual neurons in DNNs represent features.

        A prominent tool using feature visualizations is [OpenAI's Microscope](https://microscope.openai.com/models), which pairs visualizations of neurons with the dataset examples that also highly activate the neuron. Our SVD feature visualizations is a similar tool. However, our definition of "neurons," the basic unit of analysis for defining features that we are visualizing, differs from existing approaches.
        ### SVD Neuron
        Why do we want to represent the activations in a new basis? Performing interpretability analysis on only the activations is sometimes misleading. [Polysemantic neurons](https://distill.pub/2020/circuits/zoom-in/), neurons that respond to many unrelated inputs, are one prominent problem for feature visualization.

        The approach we have taken is to project the activations of a hidden layer onto the basis of eigenvectors of the weights for the layer.
        """
        st.write(body_svd)
        neuron_img = Image.open("data/svd_neuron_viz.png")
        st.image(
            neuron_img,
        )
        body_tda1 = """
        ## 2 TDA Visualizations
        ### Topological Data Analysis
        Topological Data Analysis (TDA) is a powerful tool for the analysis of large metrizable spaces of data. We explore the use of TDA to analyze the structure of profile graphs and uncover meaning behind the interconnection of the synapses, independent of labels on nodes and synapses. To accomplish this, we define a metric space on the vertices of a profile graph, which we can then analyze using persistent homology.

        """
        st.write(body_tda1)
        pipeline_img = Image.open("data/pipelineimg.png")
        st.image(pipeline_img)

        body_tda2 = """
        #### Metric Space
        The vertices of the profile graph can be represented in a metric space by constructing the distance matrix using the shortest path distance. Optionally, some kernel function can then be applied to the distances to produce a desired effect on the metric space. One example that we have explored is the Gaussian kernel, given by $g(x) = 1 - e^{-x/2\sigma}$, where $\sigma$ is the standard deviation of the finite shortest path distances. The Gaussian kernel is an increasing function that spreads out low distances and contracts high distances. When the edge weights of a profile graph are defined according to an inverted weighting scheme, the low distances correspond to the most influential connections. In this case, spreading out the low distances can reveal more nuanced structures that emerge at those distance thresholds.

        #### Persistent Homology
        Persistent homology allows us to summarize the "shape" of profile graph data based on the appearance of topological features at different distance thresholds. We calculate the persistent homology of a metric space, and then study its persistence diagram to identify topological features of the corresponding profile graph. Persistence diagrams allow us to visualize the persistence of features by plotting a point for each topological feature, whose coordinates are $(birth, death)$. The  $birth$ of a feature, such as an open loop, represents the distance threshold when the loop was formed, and the  $death$ represents the distance threshold when the loop was closed or triangulated.  For a more in depth introduction to persistent homology, [A User’s Guide to Topological Data Analysis](https://learning-analytics.info/index.php/JLA/article/view/5196) by Elizabeth Munch gives an overview of modern TDA methods, including persistent homology (Section 3).

        #### Persistence Images

        Persistence images are finite-dimensional vector representations of persistence diagrams, proposed by Adams et. al. in [Persistence Images: A Stable Vector Representation of
        Persistent Homology](https://www.jmlr.org/papers/volume18/16-337/16-337.pdf). We have used persistence images as part of our initial exploration of the topological features of profile graphs, since they provide alternative visualizations that can be compared by Euclidean distances (a metric that is much more computationally efficient than the current standard methods for comparing persistence diagrams).
        """
        st.write(body_tda2)
        pd_pi_img = Image.open("data/PDtoPI.png")
        st.image(pd_pi_img)
        body_tda3 = """
        ### TDA Visualization Tool
        Our TDA visualization tool allows persistence diagrams and persistence images to be viewed alongside the input image from which their corresponding profile graph was generated by Deep Data Profiler. The tool includes image and persistence data for 50 images from each class of the ImageNet1k dataset, profiled using element-wise and channel-wise neuron definitions, on both VGG16 and ResNet-18 architectures. All persistence images were generated using the same scale and parameters, so they can be visually compared between different input images and classes.
        """
        st.write(body_tda3)

    elif active_tab == "SVD Feature Visualizations":
        st.subheader("Singular value feature visualizations")
        with st.spinner("Loading image paths..."):
            img_paths = load_paths()

        layers = list(img_paths.keys())
        # Add a selectbox to the sidebar:
        layer_selectbox = st.sidebar.selectbox(
            "Choose a hidden layer of VGG-16 to view", layers, index=10
        )

        features = sorted(list(img_paths[layer_selectbox].keys()))

        # Add a selectbox to the sidebar:
        feature_selectbox = st.sidebar.selectbox(
            "Choose an SVD neuron to view", features, index=7
        )

        st.sidebar.subheader("SVD feature visualization")
        feature_viz = st.sidebar.empty()

        feature_option = st.sidebar.selectbox(
            label="Use pre-computed or compute on the fly",
            options=("Pre-computed", "Compute"),
        )

        # NOTE: in future, allow user to supply relative path
        feature_array = np.array([])
        if feature_option == "Pre-computed":
            relative_feature_root = Path("vgg16_imagenet_svd_average/")

            feature_name_path = Path(layer_selectbox) / Path(
                str(layer_selectbox)
                + "_"
                + str(feature_selectbox)
                + "th_singular.pkl"
            )
            feature_path = relative_feature_root / feature_name_path
            try:
                # feature_array = pkl.load(open(feature_path, "rb"))
                feature_array = read_pickle_file(str(feature_path))
            except (FileNotFoundError, ClientError):
                st.header("SVD feature visualization")
                st.write(
                    "Singular value feature visualizations not yet computed for the final two classification layers"
                )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_model().to(device)
            svd_dict_visualization = load_svd_dicts(model)
            with st.spinner("Computing feature visualization..."):
                feature_array = svd_visualization(
                    model,
                    svd_dict_visualization,
                    str(layer_selectbox),
                    feature_selectbox,
                )

        if feature_array.any():
            feature_viz.image(feature_array, width=256)

        # path_str = st.sidebar.text_input(
        #     "Relative path to ImageNet data",
        #     value="imagenet/validation",
        #     help="File path to data. For now, we support the ImageNet validation set.",
        # )
        # relative_path = Path(path_str)
        relative_path = Path("imagenet/validation")

        col_slider, _ = st.beta_columns((2, 1))
        topimages = col_slider.slider(
            "Number of top activated images to display",
            min_value=0,
            max_value=100,
            value=9,  # default value
        )
        images = img_paths[layer_selectbox][feature_selectbox]
        images = list(OrderedDict.fromkeys(images))
        top_imgpaths = images[:topimages]

        st.subheader("Top activated images")

        # Top image columns
        col1, col2, col3 = st.beta_columns(3)
        for (
            (img_path1, score1),
            (img_path2, score2),
            (img_path3, score3),
        ) in grouper(top_imgpaths, 3, ("", 0)):
            try:
                relative_img_path = relative_path / img_path1
                img1 = read_image_file(str(relative_img_path)).convert("RGB")
                col1.image(img1)

                relative_img_path = relative_path / img_path2
                img2 = read_image_file(str(relative_img_path)).convert("RGB")
                col2.image(img2)

                relative_img_path = relative_path / img_path3
                img3 = read_image_file(str(relative_img_path)).convert("RGB")
                col3.image(img3)
            except (IsADirectoryError, ClientError):
                pass

    elif active_tab == "TDA Visualizations":
        st.subheader("Persistent homology visualizations")
        neurons = st.sidebar.radio(
            "Neuron type", options=("elements", "channels")
        )
        model = "vgg16"  # hard-coded for now

        names_to_numbers, numbers_to_folders = load_class_labels_dicts()

        cls_names = st.multiselect(
            "Choose a class to view",
            list(names_to_numbers.keys()),
            default=["tench, Tinca tinca", "volcano"],
        )
        # col_slider, _ = st.beta_columns((2, 1))
        num_images = st.slider(
            "Number of images to display per class",
            min_value=1,
            max_value=50,
            value=1,  # default value
        )

        # path_str = st.sidebar.text_input(
        #     "Relative path to ImageNet data",
        #     value="imagenet/validation",
        #     help="For now, we support the ImageNet validation set.",
        # )
        # relative_path = Path(path_str)
        relative_path = Path("imagenet/validation")

        for cls_name in cls_names:
            cls_nmbr = names_to_numbers[cls_name]
            cls = numbers_to_folders[int(cls_nmbr)]
            tdapath = Path(neurons + "_0.1TH")
            pimpath = tdapath / Path("persistence_images/H1")

            img_paths = relative_path / Path(cls)
            imgnames = list_image_files(str(img_paths))
            col_tda_image, col_tda_pers, col_tda_pers_heat = st.beta_columns(3)
            col_tda_image.subheader(f"Image \n class: {cls_name}")
            col_tda_pers.subheader("Persistence Diagram")
            col_tda_pers_heat.subheader("Persistence Image")

            for img in range(num_images):
                imgname = imgnames[img]
                pimname = f"{cls}/{imgname[:-5]}_pimg.p"

                (
                    col_tda_image,
                    col_tda_pers,
                    col_tda_pers_heat,
                ) = st.beta_columns(3)

                tda_img1 = read_image_file(imgname).convert("RGB")
                col_tda_image.image(
                    tda_img1,
                )

                pdpath = f"{tdapath}/ripsers/{cls}"
                stripped_imgname = imgname.split("/")[-1]
                pdname = f"{stripped_imgname[:-5]}_ripser-invWspK.p"
                rips = read_pickle_file(f"{pdpath}/{pdname}")

                diagram = plot_diagrams(
                    rips["dgms"],
                    size=25,
                )
                st.set_option("deprecation.showPyplotGlobalUse", False)
                col_tda_pers.pyplot(diagram)

                # pimgr = pkl.load(open(f"{pimpath}/pimager.p", "rb"))
                pimgr = read_pickle_file(f"{pimpath}/pimager.p")
                dgms = rips["dgms"][1][np.isfinite(rips["dgms"][1][:, 1])]
                pim = pimgr.transform(dgms, skew=True)
                pimgr_diagram = pimgr.plot_image(pim).figure
                col_tda_pers_heat.pyplot()
    else:
        st.write("Failure")
