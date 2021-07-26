python export_model.py /data/FLARE21/models/full_runs/nano_segmenter_192/fold1/ \
                        compiled_model_nano \
                        --check_model True \
                        --check_output True \
                        --quantize False \
                        --calibration_data /data/FLARE21/training_data_192_sameKidneys/scaled_ims/