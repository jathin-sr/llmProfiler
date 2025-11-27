.PHONY: quick-profile full-profile plots plot1 help

quick-profile: # Quick run
	python profile_training.py --config config/profiling/quick_test.yaml

full-profile: # Full grid run
	python profile_training.py --config config/profiling/full_grid.yaml

plots: # make all plots
	python utils/visualization/plot_power.py
	python utils/visualization/plot_energy.py 
	python utils/visualization/plot_throughput.py
	python utils/visualization/plot_flops_per_watt.py
	python utils/visualization/plot_training_time.py
	python utils/visualization/plot_analysis.py
	python utils/visualization/plot_timing.py
	python utils/visualization/plot_special.py

plot1:
	python utils/visualization/plot_timing.py

help:
	@echo "Available targets:"
	@echo "  quick-profile    - Run quick profiling test"
	@echo "  full-profile     - Run full grid search profiling"