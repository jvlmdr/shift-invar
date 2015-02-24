cd run && \
../params \
	-train-dataset inria \
	-train-dataset-spec '{"Dir": "'$DATASETS'/inria-person/", "Set": "Train", "ExclNegTest": true}' \
	-test-dataset inria \
	-test-dataset-spec '{"Dir": "'$DATASETS'/inria-person/", "Set": "Test", "ExclNegTest": true}' \
	-pad=18 -reject-aspect=1.5 -resize-for=height \
	-max-train-scale=2 -flip -train-interp=1 \
	-margin=18 -pyr-step=1.07 -local-max -max-test-scale=1 -test-interp=1 \
	-min-match=0.5 -min-ignore=0.5 -dets-per-im=64 \
	-train.flags '-l mem=4g,walltime=8:00:00' \
	-test.flags '-l mem=1g,walltime=1:00:00' \
	-dstrfn.debug \
	../params.json
