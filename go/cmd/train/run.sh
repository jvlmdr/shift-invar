./train \
	-feat '{"Name": "hog", "Spec": {"Conf": {"Angles": 9, "CellSize": 4}}}' \
	-dataset inria -dataset-spec '{"Dir": "'$DATASETS'/inria-person/", "Set": "Train"}' \
	-width=32 -height=96 -pad=18 -reject-aspect=1.3 -resize-for=height -max-scale=2 -flip
