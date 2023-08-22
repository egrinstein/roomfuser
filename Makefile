train:
	@cd src && python -m roomfuser logs .

submit:
	@qsub qsub.pbs

viz:
	@cd src && python visualization.py