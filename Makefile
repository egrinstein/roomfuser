train:
	@cd src && python -m roomfuser logs .

submit:
	@qsub qsub.pbs
