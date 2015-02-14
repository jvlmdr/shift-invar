./svm-params \
	-dataset inria -dataset-spec '{"Dir": "'$DATASETS'/inria-person/", "Set": "Train", "ExclNegTest": true}' \
	-pad=18 -reject-aspect=1.3 -resize-for=height \
	-max-train-scale=2 -flip -train-interp=1 \
	-train.flags '-l mem=8g,walltime=4:00:00'

#	-margin=18 -pyr-step=1.1 -local-max -max-test-scale=1 -test-interp=1 \
#	-max-iou=0.3 -min-match=0.5 -min-ignore=0.5 -dets-per-im=64
