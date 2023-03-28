#!/bin/bash
for d in */ do
	for f in *.tif do
		echo -n "${d}_${f.tif}.tif"
	done
done
