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
        self["name"] = "zero_mean_unit_variance"
        self["kwargs"] = {}
        self["kwargs"]["mode"] = mode
        self["kwargs"]["axes"] = axes
        if (mode == "fixed"):
            self["kwargs"]["mean"] = mean
            self["kwargs"]["std"] = std


class ScaleLinear(Postprocessing):
    def __init__(self, gain, axes, offset):
        self["name"] = "scale_linear"
        self["kwargs"] = {}
        self["kwargs"]["gain"] = gain
        self["kwargs"]["offset"] = offset
        self["kwargs"]["axes"] = axes


class ModelZooInput(dict):

    def __init__(self, name, axes, data_type, data_range, halo, min, step,
                 preprocessing):
        self["name"] = name
        self["axes"] = axes
        self["data_type"] = data_type
        self["data_range"] = data_range
        self["halo"] = halo
        self["shape"] = {}
        self["shape"]["min"] = min
        self["shape"]["step"] = step
        self["preprocessing"] = preprocessing


class ModelZooOutput(dict):
    def __init__(self, name, axes, offset, scale, reference_input, postprocessing):
        self["name"] = name
        self["axes"] = axes
        self["shape"] = {}
        self["shape"]["offset"] = offset
        self["shape"]["scale"] = scale
        self["shape"]["reference_input"] = reference_input
        self["postprocessing"] = postprocessing


class ModelZooWeight(dict):
    def __init__(self, weight_format, source, modelfile_name, authors, tag, tensorflowversion):
        self[weight_format] = {}
        self[weight_format]["authors"] = authors
        self[weight_format]["source"] = source
        with open(modelfile_name, "rb") as f:
            bytes = f.read()
            self[weight_format]["sha256"] = sha256(bytes).hexdigest()
        self[weight_format]["tag"] = tag
        self[weight_format]["tensorflowversion"] = tensorflowversion


class ModelZooBaseData(dict):
    def __init__(self, name, desc, cite, authors, documentation,
                 tags, dependencies, covers,
                 sample_input=['./sample_input.tif'], sample_output=['./sample_output.tif'],
                 inputs=[],
                 outputs=[],
                 weights=[], license="bsd", format_version="0.3.1",
                 language="python", framework="tensorflow", config = None):
        self.name = name
        self.description = desc
        self.timestamp = datetime.now().isoformat()
        if cite is not None:
            self.cite = cite
        self.authors = authors
        self.documentation = documentation
        self.git_repo = "https://github.com/CSBDeep/CSBDeep"
        self.tags = tags
        self.license = license
        self.format_version = format_version
        self.language = language
        self.framework = framework
        self.dependencies = dependencies
        self.covers = covers
        self.sample_input = sample_input
        self.sample_output = sample_output
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        if config is not None:
            self.config = config


def modelzoo_export(tf_model, folder_path, modelzoo_data, sample_input=None, sample_output=None,
                    zip_name=None):
    export_SavedModel(tf_model, folder_path)
    if zip_name is None:
        zip_name = folder_path + '/export.bioimage.io.model.zip'
    else:
        zip_name = Path(zip_name)
    input_file = folder_path + '/sample_input.tif'
    output_file = folder_path + '/sample_output.tif'
    imsave(input_file, sample_input)
    imsave(output_file, sample_output)
    modelzoo_data.test_inputs = [input_file]
    modelzoo_data.test_outputs = [output_file]

    yml_file = folder_path + '/model.yml'
    yaml = YAML(typ='rt')
    yaml.default_flow_style = False
    with open(yml_file, 'w', encoding='UTF-8') as outfile:
        yaml.dump(modelzoo_data, outfile)

    with ZipFile(zip_name, 'a') as myzip:
        myzip.write(yml_file, arcname=os.path.basename(yml_file))
        myzip.write(input_file, arcname=os.path.basename(input_file))
        myzip.write(output_file, arcname=os.path.basename(output_file))

    print("\nModel exported in BioImage ModelZoo format:\n%s" % str(zip_name.resolve()))
