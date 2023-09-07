train:
	@cd src && python -m roomfuser logs .

train3:
	@cd src && python3 -m roomfuser logs .

submit:
	@qsub qsub.pbs

viz:
	@cd src && python visualize_backward.py

viz-fwd:
	@cd src && python visualize_forward.py

dataset:
	@cd src && python create_rir_dataset.py

activate:
	@conda activate roomfuser

nohup:
	@nohup cd src && python3 -m roomfuser logs . > out.log &