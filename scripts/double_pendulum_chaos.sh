gpu=$1

CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial1/smooth.yaml -task chaos_specific
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial2/smooth.yaml -task chaos_specific
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial3/smooth.yaml -task chaos_specific

CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial1/base.yaml -task chaos
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial2/base.yaml -task chaos
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial3/base.yaml -task chaos
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial1/smooth.yaml -task chaos
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial2/smooth.yaml -task chaos
CUDA_VISIBLE_DEVICES="$gpu" python downstream.py -config configs/double_pendulum/trial3/smooth.yaml -task chaos