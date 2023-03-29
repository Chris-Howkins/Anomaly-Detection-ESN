#!/bin/bash
for d in */; do
    for f in $d/*; do
		printf '%s\n' "${d%/}_$(basename $f)"
        mv "$f" "${d%/}_$(basename $f)"
    done
done
