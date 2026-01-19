#!/bin/bash

export MODAL_TOKEN_ID=""
export MODAL_TOKEN_SECRET=""

cd gpu_mode
python run_modal.py --submission bioml/trimul/B200_kernels/TTT_B200.py --task trimul --gpu B200 --mode leaderboard
