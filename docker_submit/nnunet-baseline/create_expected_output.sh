#!/bin/bash

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"

export nnUNet_raw="${SCRIPTPATH}/test"
export nnUNet_preprocessed="${SCRIPTPATH}/test"
export nnUNet_results="${SCRIPTPATH}/nnunet_baseline/nnUNet_results"
nnUNetv2_predict -i "${SCRIPTPATH}/test/orig" -o "${SCRIPTPATH}/test/expected_output_nnUNet" -d 101 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerUmambaSinNorm -c 3d_fullres_bs8 -f all -chk checkpoint_best.pth