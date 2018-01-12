DEMO
====


1. Go to demo notebook folder:
	
	cd ~/git/CSBDeep_code/notebooks

2. Activate virtual python environment (needed to import csbdeep)

	source activate csbdeep


3. Start everything (jupyter, tensorboard...)

	./start_all_notebooks.sh


Notes
=====


- Current example notebooks are for denoising (tribolium), isonet (retina) and MT enhancement (tubulin). 

- They should all run as is, including training for 60 epochs that should not take longer than 10 mins.

- Each notebook gets 25% FPU memory by default, see beginning of each file to change that if needed

- tensorboard should be automatically opened at
	 
	localhost:6006

- tensorboard shows all the losses (Tab SCALARS) and sample input/output images (Tab IMAGES) for each epoch.
  The prefix "pretrained" indicates the reference losses/outputs
  
  A nice thing could be to scroll through the sucessively better outputs for each epoch (e.g. pretrained/retina)
  The tubulin case is trained probabilistically, so we there there have "mean" and "scale" as ouput (instead of "output").
 

- to remove polluted tensorboard logs (without pretrained)

	rm -rf logs/latest
 

