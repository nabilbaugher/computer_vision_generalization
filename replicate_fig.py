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
import matplotlib.pyplot as plt

class Args():
    def __init__(self, novel, bg, alpha, blur, percent_size, unaligned, plot):
        self.novel = novel
        self.bg = bg
        self.alpha = alpha
        self.blur = blur
        self.percent_size = percent_size
        self.unaligned = unaligned
        self.plot = plot
        
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

def inference(model, penult_model, transform, result_dir, alpha):
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

            
    args = Args(novel=False, bg=False, alpha=alpha, blur=0.0, percent_size='100', unaligned=False, plot='alpha')
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
        # make_plots(args)
        
def plot_bias_vs_alpha():
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
        
def run_experiments(alphas, random=False):
    model, penult_model, transform = init_model(random)
    for alpha in alphas:
        print('random: {0}, alpha: {1}'.format(random, alpha))
        result_dir = 'myresults/classification/resnet50{0}/alpha_{1}'.format('_random' if random else '', alpha)
        inference(model, penult_model, transform, result_dir, alpha)
    print('done!')

# run_experiments([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], random=False)
# run_experiments([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], random=True)
plot_bias_vs_alpha()