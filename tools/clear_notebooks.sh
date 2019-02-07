#!/usr/bin/env bash

src="../examples"
for f in `find ${src} -name "*.ipynb"`; do
	jupyter nbconvert ${f} --to notebook --ClearOutputPreprocessor.enabled=True --inplace;
done
