import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from data import SilhouetteTriplets
import probabilities_to_decision
from evaluate import *
from plot import make_plots
import numpy as np
import matplotlib.pyplot as plt
import json
from deephys import Neuron, Layer, Model, import_torch_dataset, import_test_data
from collections import defaultdict


class Args():
    def __init__(self, novel, bg, alpha, blur, percent_size, unaligned, plot, get_embeddings):
        self.novel = novel
        self.bg = bg
        self.alpha = alpha
        self.blur = blur
        self.percent_size = percent_size
        self.unaligned = unaligned
        self.plot = plot
        self.get_embeddings = get_embeddings
        
def init_model(random=False):
    # set up image transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # load pretrained resnet
    model = models.resnet50(pretrained=not random)
    model.eval()
    
    # get penultimate layer
    modules = list(model.children())[:-1]
    penult_model = nn.Sequential(*modules)
    
    return model, penult_model, transform


def run_classification(model, penult_model, transform, result_dir, alpha):
    stimuli_dir = '../stimuli-shape/style-transfer/{}'.format(alpha)

    shape_categories = sorted(['knife', 'keyboard', 'elephant', 'bicycle', 'airplane',
                            'clock', 'oven', 'chair', 'bear', 'boat', 'cat',
                            'bottle', 'truck', 'car', 'bird', 'dog'])

    shape_dict = dict.fromkeys(shape_categories)  # for storing the results
    shape_categories0 = [shape + '0' for shape in shape_categories]
    shape_dict0 = dict.fromkeys(shape_categories0)

    shape_spec_dict = dict.fromkeys(shape_categories)  # contains lists of specific textures for each shape
    for shape in shape_categories:
        shape_dict[shape] = shape_dict0.copy()
        shape_spec_dict[shape] = []

            
    args = Args(
        novel=False,
        bg=False,
        alpha=alpha,
        blur=0.0,
        percent_size='100',
        unaligned=False,
        plot='alpha',
        get_embeddings=False
    )
    
    dataset = SilhouetteTriplets(args, stimuli_dir, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
    softmax = nn.Softmax(dim=1)
    softmax2 = nn.Softmax(dim=0)

    with torch.no_grad():
        # Pass images into the model one at a time
        for batch in dataloader:
            im, name = batch
            split_name = name[0].split('-')

            shape = ''.join([i for i in split_name[0] if not i.isdigit()])
            texture = ''.join([i for i in split_name[1][:-4] if not i.isdigit()])
            texture_spec = split_name[1][:-4]
            
            output = model(im)

            soft_output = softmax(output).detach().numpy().squeeze()

            decision, class_values = mapping.probabilities_to_decision(soft_output)

            shape_idx = shape_categories.index(shape)
            texture_idx = shape_categories.index(texture)
            if class_values[shape_idx] > class_values[texture_idx]:
                decision_idx = shape_idx
            else:
                decision_idx = texture_idx
            decision_restricted = shape_categories[decision_idx]
            restricted_class_values = torch.Tensor([class_values[shape_idx], class_values[texture_idx]])
            restricted_class_values = softmax2(restricted_class_values)

            shape_dict[shape][texture_spec + '0'] = [decision, class_values,
                                                        decision_restricted, restricted_class_values]
            shape_spec_dict[shape].append(texture_spec)

        csv_class_values(shape_dict, shape_categories, shape_spec_dict, result_dir)
        calculate_totals(shape_categories, result_dir)
        calculate_proportions('resnet50', result_dir)

def get_embeddings(args, stimuli_dir, model_type, penult_model, transform, alpha, replace=False, n=-1):
    """ Retrieves embeddings for each image in a dataset from the penultimate
    layer of a given model. Stores the embeddings in a dictionary (indexed by
    image name, eg. cat4-truck3). Returns the dictionary and stores it in a json
    file (embeddings/model_type/stimuli_dir.json)

    :param args: command line arguments
    :param stimuli_dir: path of the dataset
    :param model_type: the type of model, eg. saycam, resnet50, etc.
    :param penult_model: the model with the last layer removed.
    :param transform: appropriate transforms for the given model (should match training
        data stats)
    :param replace: True if existing embeddings should be replaced.
    :param n: for use when model_type = resnet50_random or ViTB16_random. Specifies
              which particular random model to use.

    :return: a dictionary indexed by image name that contains the embeddings for
        all images in a dataset extracted from the penultimate layer of a given
        model.
    """
    try:
        os.mkdir('embeddings')
    except FileExistsError:
        pass

    try:
        os.mkdir('embeddings/{0}'.format(model_type))
    except FileExistsError:
        pass

    embedding_dir = 'embeddings/{0}/style-transfer/{1}.json'.format(model_type, alpha)

    try:
        embeddings = json.load(open(embedding_dir))
        return embeddings

    except FileNotFoundError:  # Retrieve and store embeddings
        # Initialize dictionary
        embeddings = {}

        dataset = SilhouetteTriplets(args, stimuli_dir, transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            # Iterate over images
            for idx, batch in enumerate(data_loader):
                im = batch[0]
                name = batch[1][0]

                embedding = penult_model(im)
                #embedding = embedding.cpu().numpy().squeeze()
                embedding = torch.squeeze(embedding)

                embeddings[name] = embedding.tolist()

        with open(embedding_dir, 'w') as file:
            json.dump(embeddings, file)

        return embeddings

def triplets(args, model_type, stimuli_dir, embeddings, alpha, n=-1):
    """First generates all possible triplets of the following form:
    (anchor image, shape match, texture match). Then retrieves the activations
    of the penultimate layer of a given model for each image in the triplet.
    Finally, computes either cosine similarity, dot products, or Euclidean distances:
    anchor x shape match, anchor x texture match. This determines whether the model
    thinks the shape or texture match for an anchor image is closer to the anchor and
    essentially provides a secondary measure of shape/texture bias. This function
    returns a dictionary where values are a list: position 0 contains a dataframe
    of results for a given anchor, and position 1 contains an appropriate path for
    a corresponding CSV file. The keys are the names of the anchor stimuli.

    :param args: command line arguments
    :param model_type: resnet50, saycam, etc.
    :param stimuli_dir: location of dataset
    :param embeddings: a dictionary of embeddings for each image for the given model
    :param n: for use when model_type = resnet50_random or ViTB16_random. Specifies
              which particular random model to use.

    :return: a dictionary containing a dataframe of results and a path for a CSV file for
             each anchor stimulus.
    """

    dataset = SilhouetteTriplets(args, stimuli_dir, None)

    images = dataset.shape_classes.keys()
    all_triplets = dataset.triplets_by_image
    results = {key: None for key in images}  # a dictionary of anchor name to dataframe mappings

    metrics = ['dot', 'cos', 'ed']

    columns = ['Model', 'Anchor', 'Anchor Shape', 'Anchor Texture', 'Shape Match', 'Texture Match',
               'Metric', 'Shape Distance', 'Texture Distance', 'Shape Match Closer',
               'Texture Match Closer']

    cosx = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

    for anchor in images:  # Iterate over possible anchor images
        anchor_triplets = all_triplets[anchor]['triplets']
        num_triplets = len(anchor_triplets)

        df = pd.DataFrame(index=range(num_triplets * len(metrics)), columns=columns)
        df['Anchor'] = anchor[:-4]
        df['Model'] = model_type
        if args.novel:
            df['Anchor Shape'] = dataset.shape_classes[anchor]['shape']
            df['Anchor Texture'] = dataset.shape_classes[anchor]['texture']
        else:
            df['Anchor Shape'] = dataset.shape_classes[anchor]['shape_spec']
            df['Anchor Texture'] = dataset.shape_classes[anchor]['texture_spec']

        metric_mult = 0  # Ensures correct placement of results

        for metric in metrics:  # Iterate over distance metrics
            step = metric_mult * num_triplets

            for i in range(num_triplets):  # Iterate over possible triplets
                df.at[i + step, 'Metric'] = metric

                triplet = anchor_triplets[i]
                shape_match = triplet[1]
                texture_match = triplet[2]

                df.at[i + step, 'Shape Match'] = shape_match[:-4]
                df.at[i + step, 'Texture Match'] = texture_match[:-4]

                # Get image embeddings
                anchor_output = torch.FloatTensor(embeddings[anchor])
                shape_output = torch.FloatTensor(embeddings[shape_match])
                texture_output = torch.FloatTensor(embeddings[texture_match])

                if anchor_output.shape[0] == 1:
                    anchor_output = torch.squeeze(anchor_output, 0)
                    shape_output = torch.squeeze(shape_output, 0)
                    texture_output = torch.squeeze(texture_output, 0)

                if metric == 'cos':  # Cosine similarity
                    shape_dist = cosx(anchor_output, shape_output).item()
                    texture_dist = cosx(anchor_output, texture_output).item()
                elif metric == 'dot':  # Dot product
                    shape_dist = np.dot(anchor_output, shape_output).item()
                    texture_dist = np.dot(anchor_output, texture_output).item()
                else:  # Euclidean distance
                    shape_dist = torch.cdist(torch.unsqueeze(shape_output, 0), torch.unsqueeze(anchor_output, 0)).item()
                    texture_dist = torch.cdist(torch.unsqueeze(texture_output, 0),
                                               torch.unsqueeze(anchor_output, 0)).item()

                df.at[i + step, 'Shape Distance'] = shape_dist
                df.at[i + step, 'Texture Distance'] = texture_dist

                if metric == 'ed':
                    shape_dist = -shape_dist
                    texture_dist = -texture_dist

                # Compare shape/texture results
                if shape_dist > texture_dist:
                    df.at[i + step, 'Shape Match Closer'] = 1
                    df.at[i + step, 'Texture Match Closer'] = 0
                else:
                    df.at[i + step, 'Shape Match Closer'] = 0
                    df.at[i + step, 'Texture Match Closer'] = 1

            metric_mult += 1

        results[anchor] = [df, 'myresults/triplets/{0}/alpha_{1}/{2}.csv'.format(model_type, alpha, anchor[:-4])]

    return results

def run_triplets(model, penult_model, transform, random, alpha):
    stimuli_dir = '../stimuli-shape/style-transfer/{}'.format(alpha)
    model_type = 'resnet50{}'.format('_random' if random else '')
    
    args = Args(
        novel=False,
        bg=False,
        alpha=alpha,
        blur=0.0,
        percent_size='100',
        unaligned=False,
        plot='alpha',
        get_embeddings=False
    )
    
    embeddings = get_embeddings(args, stimuli_dir, model_type, penult_model, transform, alpha,
                                    replace=args.get_embeddings)
    if args.get_embeddings:
        return
    results = triplets(args, model_type, stimuli_dir, embeddings, alpha)

    # Convert result DataFrames to CSV files
    for anchor in results.keys():
        anchor_results = results[anchor]
        df = anchor_results[0]
        path = anchor_results[1]

        df.to_csv(path, index=False)

def plot_bias_vs_alpha(classification=True):
    """Plots shape bias proportions vs. alpha value (background saliency)."""

    model_list = ['resnet50', 'resnet50_random']
    colors = ['#e274d0', '#e274d0']
    labels = ['ResNet-50', 'ResNet-50 (Random)']
    styles = ['solid', 'dashed']
    markers = ['o', 'o']

    model_dict = {key: {"0.0": 0, "0.2": 0, "0.4": 0, "0.6": 0, "0.8": 0, "1.0": 0}
                  for key in model_list}

    for i in range(len(model_list)):
        model = model_list[i]

        for alpha in model_dict[model].keys():
            prop_dir = 'myresults/classification/{0}/alpha_{1}/proportions_avg.csv'.format(model, alpha)

            props = pd.read_csv(prop_dir)
            shape_bias = props.at[0, 'Shape Match Closer']

            model_dict[model][alpha] = shape_bias

    alphas = list(model_dict[model].keys())

    plt.clf()
    plt.axhline(0.5, color='#808080', linestyle=(0, (1, 3)))  # Chance line

    for i in reversed(range(len(model_list))):
        model = model_list[i]
        plt.plot(alphas, list(model_dict[model].values()), linestyle=styles[i],
                    color=colors[i], label=labels[i], marker=markers[i],
                    markersize=7.5, markeredgecolor='black', markeredgewidth=0.5,
                    markerfacecolor=colors[i])

    plt.title("Shape Bias vs. \u03B1")
    plt.xlabel('\u03B1 (Background Texture Transparency)', fontsize=10)
    plt.ylabel('Proportion of Shape Decisions', fontsize=10)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in reversed(range(len(model_list)))],
               [labels[idx] for idx in reversed(range(len(model_list)))],
               prop={'size': 8})

    plt.tight_layout()

    plt.savefig('myfigures/bias_vs_alpha_classification.png')

    plt.clf()

def run_experiments(alphas, classification=True, random=False):
    model, penult_model, transform = init_model(random)
    # if export_to_deephys:
    #     neuronList = []
    #     for i in range(np.shape(all_activs)[1]):
    #         neuronList.append(Neuron())

    #         layerList = []
    #         layerList.append(Layer(
    #             layerID = "linear1",
    #             neurons = neuronList
    #         ))

    #     #####
    #     neuronList = []
    #     for i in range(np.shape(all_outputs)[1]):
    #         neuronList.append(Neuron())

    #         layerList.append(Layer(
    #             layerID = "classification",
    #             neurons = neuronList
    #         ))
    #     #####

    #     model = Model(
    #         name = "colored_MNIST",
    #         suffix = None,
    #         layers = layerList
    #     )

    #     model.save()
        
    if classification:
        for alpha in alphas:
            print('random: {0}, alpha: {1}'.format(random, alpha))
            result_dir = 'myresults/classification/resnet50{0}/alpha_{1}'.format('_random' if random else '', alpha)
            run_classification(model, penult_model, transform, result_dir, alpha)
    else:
        for alpha in alphas:
            print('random: {0}, alpha: {1}'.format(random, alpha))
            result_dir = 'myresults/triplets/resnet50{0}/alpha_{1}'.format('_random' if random else '', alpha)
            run_triplets(model, penult_model, transform, random, alpha)
    print('all done!')

# get test data for deephys
def get_test_data(testloader):
    print('getting test data')
    all_images = []
    all_cats = []
    resize = transforms.Resize((32,32)) #default size is 32 for deephys
    
    with torch.no_grad():
        for batch in testloader:
            im, name = batch
            
            #Resize data for deephys
            resized_im = resize(im)
            all_images.append(resized_im)
            all_cats.append(name)
    
    return all_images, all_cats

# get activations and outputs for deephys
def get_activations_and_outputs(model, penult_model, dataloader):
    print('getting activations and outputs')
    # Define a dictionary to store the activations and outputs
    activation = {}

    # Define a function to create a hook that stores the output of a given layer
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register hooks on the penultimate layer and the final layer
    penult_model.register_forward_hook(get_activation('penult'))
    model.register_forward_hook(get_activation('output'))
    
    # Initialize lists to store the activations and outputs
    all_activs = []
    all_outputs = []

    # Pass images through the model one at a time
    with torch.no_grad():
        for batch in dataloader:
            im, name = batch

            # Pass the image through the penultimate layer
            penult_output = penult_model(im)

            # Pass the output of the penultimate layer through the final layer
            model_output = model(im)

            # Store the activations and outputs in the activation dictionary
            activation['penult'] = penult_output.detach()
            activation['output'] = model_output.detach()

            # Append the activations and outputs to the lists
            all_activs.append(activation['penult'])
            all_outputs.append(activation['output'])

    # Concatenate the activations and outputs into single tensors
    all_activs = torch.cat(all_activs)
    all_outputs = torch.cat(all_outputs)

    return all_activs, all_outputs


def export_to_deephys(alphas, random=False):
    model, penult_model, transform = init_model(random)
    
    for alpha in alphas:
        stimuli_dir = '../stimuli-shape/style-transfer/{}'.format(alpha)
                
        args = Args(
            novel=False,
            bg=False,
            alpha=alpha,
            blur=0.0,
            percent_size='100',
            unaligned=False,
            plot='alpha',
            get_embeddings=False,
        )
        
        dataset = SilhouetteTriplets(args, stimuli_dir, transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Get test data
        all_images, all_cats = get_test_data(dataloader)
        
        # if activations and outputs already exist, load them
        if os.path.exists('myresults/deephys/resnet50{0}/alpha_{1}_activations.pt'.format('_random' if random else '', alpha)):
            all_activs = torch.load('myresults/deephys/resnet50{0}/alpha_{1}_activations.pt'.format('_random' if random else '', alpha))
            all_outputs = torch.load('myresults/deephys/resnet50{0}/alpha_{1}_outputs.pt'.format('_random' if random else '', alpha))
        else:
            # Get the activations and outputs
            all_activs, all_outputs = get_activations_and_outputs(model, penult_model, dataloader)
            
            # save activations and outputs
            torch.save(all_activs, 'myresults/deephys/resnet50{0}/alpha_{1}_activations.pt'.format('_random' if random else '', alpha))
            torch.save(all_outputs, 'myresults/deephys/resnet50{0}/alpha_{1}_outputs.pt'.format('_random' if random else '', alpha))
        
        #@title Save the model file
        neuronList = []
        for i in range(np.shape(all_activs)[1]):
            neuronList.append(Neuron())
            
        layerList = []
        layerList.append(Layer(
            layerID = "linear1",
            neurons = neuronList
        ))

        #####
        neuronList = []
        for i in range(np.shape(all_outputs)[1]):
            neuronList.append(Neuron())
        print('neuron list length', len(neuronList))


        layerList.append(Layer(
            layerID = "classification",
            neurons = neuronList
        ))
        #####

        model = Model(
            name = "resnet50{0}_alpha_{1}".format('_random' if random else '', alpha),
            suffix = None,
            layers = layerList
        )

        model.save()

        mean=torch.tensor([0.485, 0.456, 0.406])
        std=torch.tensor([0.229, 0.224, 0.225])
        print(set(all_cats))
        
        shape_categories = sorted(['knife', 'keyboard', 'elephant', 'bicycle', 'airplane',
                                   'clock', 'oven', 'chair', 'bear', 'boat', 'cat',
                                   'bottle', 'truck', 'car', 'bird', 'dog'])
        all_cats_dummy = [0 for i in range(len(all_cats))]
        print(all_images[0][0][0][0][0])
        transformed_images = [torch.permute(torch.squeeze(image),[1,2,0])*std + mean for image in all_images]
        print(type(transformed_images[0][0][0][0]))
        test = import_test_data(
            "StylizedImagenet",
            shape_categories,
            [all_activs, all_outputs],
            model,
            transformed_images,
            all_cats_dummy,
        )
        test.suffix = None
        test.save()
        
        
    
# run_experiments([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], classification=True, random=False)
# run_experiments([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], classification=True, random=True)
# run_experiments([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], classification=False, random=False)
# run_experiments([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], classification=False, random=True)
# plot_bias_vs_alpha()

export_to_deephys([0.0], random=False)