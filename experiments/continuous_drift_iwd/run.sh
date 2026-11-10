#!/bin/sh
# Full-size RWD EV FastPlant, 20 Hz control / 100 Hz physics, pedals interface.
# The legacy XCAR wheel-speed interface is kept as a second mode for A/B runs
# (--control-mode wheel_speed) and the original plant as --plant iwd.
set -e
EXP=${EXP:-ev_pedals_v1}
CAR=${CAR:-fullsize_racecar_ev}

python ../../run.py train continuous_drift_iwd \
    --car-preset $CAR --num-parallel 16384 --horizon 128 \
    --gamma 0.999 --tau 0.97 --epochs 300 \
    --plant fast --control-mode pedals --exp-name $EXP

python ../../run.py test continuous_drift_iwd --car-preset $CAR \
    --plant fast --control-mode pedals --exp-name $EXP --ref-mode hybrid
python plot.py --filename hybrid_ref_exp
python ../../run.py test continuous_drift_iwd --car-preset $CAR \
    --plant fast --control-mode pedals --exp-name $EXP --ref-mode circle
python plot.py --filename circle_ref_exp
python ../../run.py test continuous_drift_iwd --car-preset $CAR \
    --plant fast --control-mode pedals --exp-name $EXP --ref-mode eight
python plot.py --filename eight_ref_exp
