gpu=$1

CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/spring_mass/trial4/regress-smooth-filtered.yaml -task all
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/spring_mass/trial4/regress-base-filtered.yaml -task all
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/single_pendulum/trial1/regress-smooth-filtered.yaml -task all
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/single_pendulum/trial1/regress-base-filtered.yaml -task all
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial2/regress-smooth-filtered.yaml -task all
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial2/regress-base-filtered.yaml -task all
