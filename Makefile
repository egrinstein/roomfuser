train:
	@python -m roomfuser logs

train3:
	@python3 -m roomfuser logs

submit:
	@qsub qsub.pbs

viz:
	@python visualize_backward.py

viz-fwd:
	@python visualize_forward.py

dataset:
	@python create_rir_dataset.py

activate:
	@conda activate roomfuser

nohup:
	@nohup python3 -m roomfuser logs > out.log &