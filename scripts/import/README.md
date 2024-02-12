These scripts import and sort the dicom data.

Typical usage:

1. import_from_rico
2. sort_dicoms
3. compress_dicoms

The first of these , `import_from_rico`, expects a tsv file of the form:
SUBJECT SCAN_1_DIR_NAME SCAN_2_DIR_NAME on the remote system

It then rsyncs the data to the cluster, and sorts the dicoms in a pseudo-BIDS structure

Next, `sort_dicoms` loops through each subject/scan, and creates subfolders for each session

This will look like:
`sub-1/scan-1/01_Localizer`

Finally, `compress_dicoms` compresses each of the subfolders into a named .tar.gz

This will look like:
`sub-1/scan-1/01_Localizer.tar.gz`
