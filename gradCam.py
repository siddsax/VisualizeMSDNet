import torch
import pdb
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask, name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite(name +"_cam.jpg", np.uint8(255 * cam))


# class FeatureExtractor():
#     """ Class for extracting activations and 
#     registering gradients from targetted intermediate layers """
#     def __init__(self, model, target_layers):
#         self.model = model
#         self.target_layers = target_layers
#         self.gradients = []

#     def save_gradient(self, grad):
#     	self.gradients.append(grad)

#     def __call__(self, x):
#         outputs = []
#         self.gradients = []
#         print("===")
#         for name, module in self.model._modules.items():
#             x = module(x)
#             print(name)
#             if name in self.target_layers:
#                 x.register_hook(self.save_gradient)
#                 outputs += [x]
#         exit()
#         return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feats = feats
		self.grads = grads
		self.output = output
		# self.feature_extractor = FeatureExtractor(self.model.subnets, target_layers) #############

	def get_gradients(self):
		return self.grads

	def __call__(self):
		# target_activations, output  = self.feature_extractor(x)
		# output = output.view(output.size(0), -1)
		# output = self.model.classifier(output)
		return self.feats, self.output


class GradCam:
	def __init__(self, model):
		self.model = model
		self.model.eval()
		self.cuda = torch.cuda.is_available()
		if self.cuda:
			self.model = model.cuda()

		# self.extractor = ModelOutputs(self.model, feats, grads, output)

	def forward(self, input):
		return self.model(input) 

	# def __call__(self, index = None, features=None, scores=None):
	def __call__(self, index = None, input_var=None, imgNo=0, ClfrNo=0):

		# Getting output of model right here
		scores, features = self.model(input_var, 0.0, p=1)
		# Choosing the classifier number 
		scores, features = scores[ClfrNo][0], features[ClfrNo][0]

		# Get heatmap of predicted class if none
		if index == None:
			index = np.argmax(scores.cpu().data.numpy())

		# one_hot sets the index as 1 to get the gradients wrt that class
		one_hot = np.zeros((1, scores.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * scores)
		else:
			one_hot = torch.sum(one_hot * scores)

		self.model.subnets.zero_grad()
		# backward pass to get gradients		
		one_hot.backward(create_graph=True)#retain_variables=True)
		grads_val = self.model.gradients[-1].cpu().data.numpy()

		target = features#[-1]
		target = target.cpu().data.numpy()#[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		# weighted summation
		for i, w in enumerate(weights):
			cam += w * target[i]#, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (512, 512))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLU(Function):

	def forward(self, input):
		positive_mask = (input > 0).type_as(input)
		output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
		self.save_for_backward(input, output)
		return output

	def backward(self, grad_output):
		input, output = self.saved_tensors
		grad_input = None

		positive_mask_1 = (input > 0).type_as(grad_output)
		positive_mask_2 = (grad_output > 0).type_as(grad_output)
		grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

		return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model):
		self.model = model
		self.model.eval()
		self.cuda = torch.cuda.is_available()
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.subnets._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, index = None, input_var=None, imgNo=0, ClfrNo=0):

		output= self.model(input_var, 0.0)[ClfrNo][imgNo]
		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.subnets.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(create_graph=True)

		output = input_var.grad.cpu().data.numpy()
		output = output[0,:,:,:].transpose(1,2,0)
		output = cv2.resize(output, (512, 512))
		return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
						help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./examples/both.png',
						help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
		print("Using GPU for acceleration")
	else:
		print("Using CPU for computation")

	return args

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()

	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	grad_cam = GradCam(model = models.vgg19(pretrained=True), \
					target_layer_names = ["35"], use_cuda=args.use_cuda)

	img = cv2.imread(args.image_path, 1)
	img = np.float32(cv2.resize(img, (224, 224))) / 255
	input = preprocess_image(img)

	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
	target_index = None

	mask = grad_cam(input, target_index)

	show_cam_on_image(img, mask)

	gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=args.use_cuda)
	gb = gb_model(input, index=target_index)
	utils.save_image(torch.from_numpy(gb), 'gb.jpg')

	cam_mask = np.zeros(gb.shape)
	for i in range(0, gb.shape[0]):
		cam_mask[i, :, :] = mask

	cam_gb = np.multiply(cam_mask, gb)
	utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
