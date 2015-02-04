./train \
	-feat '{"Name": "hog", "Spec": {"Conf": {"Angles": 9, "CellSize": 4}}}' \
	-dataset inria -dataset-spec '{"Dir": "'$DATASETS'/inria-person/", "Set": "Train", "ExclNegTest": true}' \
	-width=32 -height=96 -pad=18 -reject-aspect=1.3 -resize-for=height \
	-max-train-scale=2 -flip -train-interp=1 \
	-max-neg-train-num=0 -max-neg-train-frac=0.2 \
	-gamma=0.5 -lambda=1 -epochs=8 \
	-margin=18 -pyr-step=1.1 -local-max -max-test-scale=1 -test-interp=1 \
	-max-iou=0.3 -min-match=0.5 -min-ignore=0.5 -dets-per-im=64
