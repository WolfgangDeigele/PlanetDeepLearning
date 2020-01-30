'''
Copyright 2018 Esri
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.​
'''

import importlib
import json
import os
import sys
import arcpy
import numpy as np
import cv2

sys.path.append(os.path.dirname(__file__))
# from attribute_table import attribute_table
#import prf_utils

import importlib
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from keras.models import model_from_json
from keras.metrics import mean_squared_error

class Unet:
	def initialize(self, model, model_as_file):
	
		# clean the background session
		K.clear_session()

		# load the emd file into a dictionary
		if model_as_file:
			with open(model, 'r') as f:
				self.json_info = json.load(f)
		else:
			self.json_info = json.loads(model)

		# get the path to the trained model
		model_path = self.json_info['ModelFile']

		# load the trained model
		self.model = load_model(model_path, custom_objects={'weighted_binary_crossentropy': 'binary_crossentropy', 'mean_iou': mean_squared_error})

		# build a default background tensorflow computational graph
		self.graph = tf.get_default_graph()
		
	def getParameterInfo(self, required_parameters):
		return required_parameters

	def getConfiguration(self, **scalars):
		print(scalars)
		self.padding = int(scalars['padding'])
		
		#get the threshold parameter
		self.threshold = float(scalars['threshold'].replace(",", "."))

		return {
			'extractBands': tuple(self.json_info['ExtractBands']),
			# padding should be 0
			'padding': int(scalars['padding']),
			'tx': self.json_info['ImageWidth'] - 2 * self.padding,
			'ty': self.json_info['ImageHeight'] - 2 * self.padding
		}

class ChildImageClassifier(Unet):
	# this class extends from the Unet class, which has all the fields and methods from Unet class
	def updatePixels(self, tlc, shape, props, **pixelBlocks):
		# get the input image tile
		image = pixelBlocks['raster_pixels']
		# get the shape of input image, which was specified in the emd file
		_, height, width = image.shape
		# transpose the image from [bands, height, width] to [height, width, bands]
		#image = np.transpose(image, [1,2,0])
		# create a image holder
		#image_ = np.zeros(image.shape)
		# first 10 bands are Sentinel, which should be divided by 10000 to get the reflectance
		#image_[:,:,:10] = image[:,:,:10] / 10000.
		# first rest bands are ASTER, which should be divided by 1000 to get the reflectance
		#image_[:,:,10:] = image[:,:,10:] / 1000.
		# expand image into shape of [batch_size, bands, height, width]
		# here we have only 1 batch
		
		image_in = np.moveaxis(image, 0, 2)
		#print(image_in.shape)
		
		#for c in range(0, 4):
		#	image_in[:,:,c] = cv2.equalizeHist(image_in[:,:,c])
		image_in = image_in / 65536
		image_in = np.expand_dims(image_in, axis=0)
		#image_in = np.expand_dims(image, axis=0)

		with self.graph.as_default():
			# get the predicted probability of the batch (here, only one batch)
			# results should be in shape of [height, width, classes]
			results = self.model.predict(image_in, verbose = 0)

		# get the raw predicted classes with the largest probability
		#preds = np.argmax(results, axis=-1)
		# get the largest probability at each pixel
		#probs = np.max(results, axis=-1)
		# select the pixels which largest probability is less than 0.5
		# and assign those pixels to the uncertain class		
		#uncertain = np.where(probs < 0.5)
		#preds[uncertain] = 1
		
		#TODO Größer/Kleiner Zeichen umdrehen, so ist es eigentlich falsch!
		results[results >= self.threshold] = 1
		results[results < self.threshold] = 0
		#print(results)
		
		# increase the class by 1, start from 1, instead of 0
		#preds = preds + 1
		#return image[0,:,:]
		#results = np.moveaxis(results, 2, 0)
		print(results.shape)
		return results[:,:,:,0].astype(np.uint8)


class PlanetClassifier:
	def __init__(self):
		self.name = 'Image Classifier'
		self.description = 'Image classification python raster function to inference a tensorflow ' \
						   'deep learning model'

	def initialize(self, **kwargs):
		if 'model' not in kwargs:
			return

		model = kwargs['model']
		model_as_file = True
		try:
			with open(model, 'r') as f:
				self.json_info = json.load(f)
		except FileNotFoundError:
			try:
				self.json_info = json.loads(model)
				model_as_file = False
			except json.decoder.JSONDecodeError:
				raise Exception("Invalid model argument")

		sys.path.append(os.path.dirname(__file__))
		framework = self.json_info['Framework']

		if 'device' in kwargs:
			device = kwargs['device']
			if device < -1:
				os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
				device = prf_utils.get_available_device()
			os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
		else:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
			
		
		self.percent = 0

		# initialize the child_image_classifier which was implemented above 
		self.child_image_classifier = ChildImageClassifier()
		self.child_image_classifier.initialize(model, model_as_file)

	def getParameterInfo(self):
		required_parameters = [
			{
				'name': 'raster',
				'dataType': 'raster',
				'required': True,
				'displayName': 'Raster',
				'description': 'Input Raster'
			},
			{
				'name': 'model',
				'dataType': 'string',
				'required': True,
				'displayName': 'Input Model Definition (EMD) File',
				'description': 'Input model definition (EMD) JSON file'
			},
			{
				"name": "threshold",
				"dataType": "float",
				"value": 0.5,
				"required": True,
				"displayName": "Threshold",
				"description": "Validation Threshold"
			},
			{
				'name': 'device',
				'dataType': 'numeric',
				'required': False,
				'displayName': 'Device ID',
				'description': 'Device ID'
			},
		]

		if 'ModelPadding' not in self.json_info:
			required_parameters.append(
				{
					'name': 'padding',
					'dataType': 'numeric',
					'value': 0,
					'required': False,
					'displayName': 'Padding',
					'description': 'Padding'
				},
			)

		if 'BatchSize' not in self.json_info:
			required_parameters.append(
				{
					'name': 'batch_size',
					'dataType': 'numeric',
					'required': False,
					'value': 1,
					'displayName': 'Batch Size',
					'description': 'Batch Size'
				},
			)

		return self.child_image_classifier.getParameterInfo(required_parameters)

	def getConfiguration(self, **scalars):
		configuration = self.child_image_classifier.getConfiguration(**scalars)
		if 'DataRange' in self.json_info:
			configuration['dataRange'] = tuple(self.json_info['DataRange'])
		configuration['inheritProperties'] = 2|4|8
		configuration['inputMask'] = True
		return configuration

	def updateRasterInfo(self, **kwargs):
		kwargs['output_info']['bandCount'] = 1
		kwargs['output_info']['pixelType'] = 'u2'
		class_info = self.json_info['Classes']
		attribute_table = {'features': []}
		for i, c in enumerate(class_info):
			attribute_table['features'].append(
				{
					'attributes':{
						'OID':i+1,
						'Value':c['Value'],
						'Class':c['Name'],
						'Red':c['Color'][0],
						'Green':c['Color'][1],
						'Blue':c['Color'][2]
					}
				}
			)
		kwargs['output_info']['rasterAttributeTable'] = json.dumps(attribute_table)

		return kwargs

	def updatePixels(self, tlc, shape, props, **pixelBlocks):
		# set pixel values in invalid areas to 0
		raster_mask = pixelBlocks['raster_mask']
		raster_pixels = pixelBlocks['raster_pixels']
		raster_pixels[np.where(raster_mask == 0)] = 0
		pixelBlocks['raster_pixels'] = raster_pixels

		# call the child_image_classifier.updatePixels to get the prediction
		pixelBlocks['output_pixels'] = self.child_image_classifier.updatePixels(tlc, shape, props, **pixelBlocks).astype(props['pixelType'], copy=False)
		
		#arcpy.SetProgressor("step", "Copying shapefiles to geodatabase...", 1, 20, 1)
		#arcpy.SetProgressorPosition(3)
		#self.percent = self.percent + 1
		#arcpy.SetProgressorPosition(self.percent)
		#print("progress changed: " + str(self.percent))
		
		return pixelBlocks