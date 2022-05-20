# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division

from collections import OrderedDict
from warnings import warn
from ..utils import _raise
from ..utils.six import Path

from ..utils.tf import keras_import
get_file = keras_import('utils', 'get_file')


_MODELS = {}
_ALIASES = {}


def clear_models_and_aliases(*cls):
    if len(cls) == 0:
        _MODELS.clear()
        _ALIASES.clear()
    else:
        for c in cls:
            if c in _MODELS:  del _MODELS[c]
            if c in _ALIASES: del _ALIASES[c]


def register_model(cls, key, url, hash):
    """ Example: 
    
    register_model(StarDist2D,   'my_great_model', 'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', md5sum_as_astring)

    """
    # key must be a valid file/folder name in the file system
    models = _MODELS.setdefault(cls,OrderedDict())
    key not in models or warn("re-registering model '%s' (was already registered for '%s')" % (key, cls.__name__))
    models[key] = dict(url=url, hash=hash)


def register_aliases(cls, key, *names):
    # aliases can be arbitrary strings
    if len(names) == 0: return
    models = _MODELS.get(cls,{})
    key in models or _raise(ValueError("model '%s' is not registered for '%s'" % (key, cls.__name__)))
    aliases = _ALIASES.setdefault(cls,OrderedDict())
    for name in names:
        aliases.get(name,key) == key or warn("alias '%s' was previously registered with model '%s' for '%s'" % (name, aliases[name], cls.__name__))
        aliases[name] = key


def get_registered_models(cls, return_aliases=True, verbose=False):
    models = _MODELS.get(cls,{})
    aliases = _ALIASES.get(cls,{})
    model_keys = tuple(models.keys())
    model_aliases = {key: tuple(name for name in aliases if aliases[name] == key) for key in models}
    if verbose:
        # this code is very messy and should be refactored...
        _n = len(models)
        _str_model  = 'model' if _n == 1 else 'models'
        _str_is_are = 'is' if _n == 1 else 'are'
        _str_colon  = ':' if _n > 0 else ''
        print("There {is_are} {n} registered {model_s} for '{clazz}'{c}".format(
              n=_n, clazz=cls.__name__, is_are=_str_is_are, model_s=_str_model, c=_str_colon))
        if _n > 0:
            print()
            _maxkeylen = 2 + max(len(key) for key in models)
            print("Name{s}Alias(es)".format(s=' '*(_maxkeylen-4+3)))
            print("────{s}─────────".format(s=' '*(_maxkeylen-4+3)))
            for key in models:
                _aliases = '   '
                _m = len(model_aliases[key])
                if _m > 0:
                    _aliases += "'%s'" % "', '".join(model_aliases[key])
                else:
                    _aliases += "None"
                _key = ("{s:%d}"%_maxkeylen).format(s="'%s'"%key)
                print("{key}{aliases}".format(key=_key, aliases=_aliases))
    return ((model_keys, model_aliases) if return_aliases else model_keys)


def get_model_details(cls, key_or_alias, verbose=False):
    models = _MODELS.get(cls,{})
    if key_or_alias in models:
        key = key_or_alias
        alias = None
    else:
        aliases = _ALIASES.get(cls,{})
        alias = key_or_alias
        alias in aliases or _raise(ValueError("'%s' is neither a key or alias for '%s'" % (alias, cls.__name__)))
        key = aliases[alias]
    if verbose:
        print("Found model '{model}'{alias_str} for '{clazz}'.".format(
        model=key, clazz=cls.__name__, alias_str=('' if alias is None else " with alias '%s'" % alias)))
    return key, alias, models[key]


def get_model_folder(cls, key_or_alias):
    key, alias, m = get_model_details(cls, key_or_alias)
    target = str(Path('models') / cls.__name__ / key)
    path = Path(get_file(fname=key+'.zip', origin=m['url'], file_hash=m['hash'],
                         cache_subdir=target, extract=True))
    assert path.exists() and path.parent.exists()
    return path.parent


def get_model_instance(cls, key_or_alias):
    path = get_model_folder(cls, key_or_alias)
    model = cls(config=None, name=path.stem, basedir=path.parent)
    model.basedir = None # make read-only
    return model
