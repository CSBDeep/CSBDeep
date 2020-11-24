import os
from datetime import datetime
from hashlib import sha256
from zipfile import ZipFile

from ruamel.yaml import YAML
from tifffile import imsave

from csbdeep.utils.six import Path
from csbdeep.utils.tf import export_SavedModel


class Preprocessing(dict):
    def __init__(self):
        pass


class Postprocessing(dict):
    def __init__(self):
        pass


class ZeroMeanUnitVariance(Preprocessing):
    def __init__(self, mode, axes, mean, std):
        self.name = "Zero_mean_unit_variance"
        self.kwargs = {}
        self.kwargs.mode = mode
        self.kwargs.axes = axes
        if (mode == "fixed"):
            self.kwargs.mean = mean
            self.kwargs.std = std


class ScaleLinear(Postprocessing):
    def __init__(self, mode, axes):
        self.name = "Zero_mean_unit_variance"
        self.kwargs = {}
        self.kwargs.gain = mode
        self.kwargs.offset = axes


class ModelZooInput(dict):

    def __init__(self, name, axes, data_type, data_range, halo, min, step,
                 preprocessing):
        self.name = name
        self.axes = axes
        self.data_type = data_type
        self.data_range = data_range
        self.halo = halo
        self.shape = {}
        self.shape.min = min
        self.shape.step = step
        self.preprocessing = preprocessing


class ModelZooOutput(dict):
    def __init__(self, name, axes, offset, scale, reference_input, postprocessing):
        self.name = name
        self.axes = axes
        self.shape = {}
        self.shape.offset = offset
        self.shape.scale = scale
        self.shape.reference_input = reference_input
        self.postprocessing = postprocessing


class ModelZooWeight(dict):
    def __init__(self, weight_format, source, modelfile_name):
        self.weight_format = weight_format
        self.source = source
        with open(modelfile_name, "rb") as f:
            bytes = f.read()
            self.sha256 = sha256(bytes).hexdigest()
        self.timestamp = datetime.now().isoformat()


class ModelZooBaseData(dict):
    def __init__(self, name, desc, cite, authors, documentation,
                 tags=None,
                 sample_input=['./sample_input.tif'], sample_output=['./sample_output.tif'],
                 inputs=[],
                 outputs=[],
                 weights=[], license="bsd", format_version="0.3.0",
                 language="python", framework="tensorflow", config = None):
        self.name = name
        self.desc = desc
        self.cite = cite
        self.authors = authors
        self.documentation = documentation
        self.tags = tags
        self.license = license
        self.format_version = format_version
        self.language = language
        self.framework = framework
        self.sample_input = sample_input
        self.sample_output = sample_output
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        self.config = config


def modelzoo_export(tf_model, folder_path, modelzoo_data, sample_input=None, sample_output=None,
                    zip_name=None):
    export_SavedModel(tf_model, folder_path)
    modelzoo_weight = ModelZooWeight("tensorflow_keras_CARE", './variables/variables', str(zip_name))
    modelzoo_data.weights = [modelzoo_weight]
    if zip_name is None:
        zip_name = folder_path / 'export.bioimage.io.model.zip'
    else:
        zip_name = Path(zip_name)
    yml_file = folder_path / "model.yml"
    yaml = YAML(typ='rt')
    yaml.default_flow_style = False
    with open(yml_file, 'w', encoding='UTF-8') as outfile:
        yaml.dump(modelzoo_data, outfile)

    input_file = folder_path / 'sample_input.tif'
    output_file = folder_path / 'sample_output.tif'
    imsave(input_file, sample_input)
    imsave(output_file, sample_output)

    with ZipFile(zip_name, 'a') as myzip:
        myzip.write(yml_file, arcname=os.path.basename(yml_file))
        myzip.write(input_file, arcname=os.path.basename(input_file))
        myzip.write(output_file, arcname=os.path.basename(output_file))

    print("\nModel exported in BioImage ModelZoo format:\n%s" % str(zip_name.resolve()))
