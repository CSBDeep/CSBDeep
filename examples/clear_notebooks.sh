for f in `find . -name "*.ipynb"`; do 
	jupyter nbconvert ${f} --to notebook --ClearOutputPreprocessor.enabled=True --inplace;
done
