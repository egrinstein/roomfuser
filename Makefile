train:
	@cd src && python -m roomfuser logs .

submit:
	@qsub qsub.pbs

viz:
	@cd src && python visualize_backward.py

viz-fwd:
	@cd src && python visualize_forward.py