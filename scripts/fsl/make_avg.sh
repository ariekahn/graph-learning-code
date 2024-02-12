#!/bin/bash
set -euo pipefail

mkdir -p avgtemp
tmpdir=$(mktemp)

fslmaths "../../derived/fsl/sub-GLS003/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS003_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS004/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS004_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS005/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS005_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS006/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS006_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS008/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS008_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS009/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS009_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS010/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS010_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS011/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS011_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS013/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS013_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS014/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS014_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS017/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS017_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS018/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS018_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS019/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS019_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS020/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS020_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS021/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS021_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS022/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS022_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS023/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS023_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS024/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS024_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS025/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS025_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS026/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS026_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS027/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS027_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS028/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS028_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS030/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS030_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS033/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS033_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS037/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS037_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS038/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS038_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS039/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS039_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS040/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS040_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS043/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS043_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS044/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS044_lh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS045/localizer.feat/stats/zstat1_masked_lh.nii.gz" -abs -bin "$tmpdir/GLS045_lh.nii.gz"

fslmaths "$tmpdir/GLS003_lh.nii.gz" \
    -add "$tmpdir/GLS004_lh.nii.gz" \
    -add "$tmpdir/GLS005_lh.nii.gz" \
    -add "$tmpdir/GLS006_lh.nii.gz" \
    -add "$tmpdir/GLS008_lh.nii.gz" \
    -add "$tmpdir/GLS009_lh.nii.gz" \
    -add "$tmpdir/GLS010_lh.nii.gz" \
    -add "$tmpdir/GLS011_lh.nii.gz" \
    -add "$tmpdir/GLS013_lh.nii.gz" \
    -add "$tmpdir/GLS014_lh.nii.gz" \
    -add "$tmpdir/GLS017_lh.nii.gz" \
    -add "$tmpdir/GLS018_lh.nii.gz" \
    -add "$tmpdir/GLS019_lh.nii.gz" \
    -add "$tmpdir/GLS020_lh.nii.gz" \
    -add "$tmpdir/GLS021_lh.nii.gz" \
    -add "$tmpdir/GLS022_lh.nii.gz" \
    -add "$tmpdir/GLS023_lh.nii.gz" \
    -add "$tmpdir/GLS024_lh.nii.gz" \
    -add "$tmpdir/GLS025_lh.nii.gz" \
    -add "$tmpdir/GLS026_lh.nii.gz" \
    -add "$tmpdir/GLS027_lh.nii.gz" \
    -add "$tmpdir/GLS028_lh.nii.gz" \
    -add "$tmpdir/GLS030_lh.nii.gz" \
    -add "$tmpdir/GLS033_lh.nii.gz" \
    -add "$tmpdir/GLS037_lh.nii.gz" \
    -add "$tmpdir/GLS038_lh.nii.gz" \
    -add "$tmpdir/GLS039_lh.nii.gz" \
    -add "$tmpdir/GLS040_lh.nii.gz" \
    -add "$tmpdir/GLS043_lh.nii.gz" \
    -add "$tmpdir/GLS044_lh.nii.gz" \
    -add "$tmpdir/GLS045_lh.nii.gz" lh.nii.gz

fslmaths "../../derived/fsl/sub-GLS003/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS003_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS004/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS004_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS005/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS005_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS006/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS006_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS008/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS008_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS009/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS009_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS010/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS010_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS011/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS011_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS013/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS013_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS014/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS014_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS017/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS017_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS018/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS018_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS019/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS019_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS020/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS020_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS021/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS021_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS022/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS022_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS023/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS023_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS024/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS024_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS025/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS025_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS026/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS026_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS027/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS027_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS028/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS028_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS030/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS030_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS033/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS033_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS037/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS037_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS038/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS038_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS039/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS039_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS040/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS040_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS043/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS043_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS044/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS044_rh.nii.gz"
fslmaths "../../derived/fsl/sub-GLS045/localizer.feat/stats/zstat1_masked_rh.nii.gz" -abs -bin "$tmpdir/GLS045_rh.nii.gz"

fslmaths "$tmpdir/GLS003_rh.nii.gz" \
    -add "$tmpdir/GLS004_rh.nii.gz" \
    -add "$tmpdir/GLS005_rh.nii.gz" \
    -add "$tmpdir/GLS006_rh.nii.gz" \
    -add "$tmpdir/GLS008_rh.nii.gz" \
    -add "$tmpdir/GLS009_rh.nii.gz" \
    -add "$tmpdir/GLS010_rh.nii.gz" \
    -add "$tmpdir/GLS011_rh.nii.gz" \
    -add "$tmpdir/GLS013_rh.nii.gz" \
    -add "$tmpdir/GLS014_rh.nii.gz" \
    -add "$tmpdir/GLS017_rh.nii.gz" \
    -add "$tmpdir/GLS018_rh.nii.gz" \
    -add "$tmpdir/GLS019_rh.nii.gz" \
    -add "$tmpdir/GLS020_rh.nii.gz" \
    -add "$tmpdir/GLS021_rh.nii.gz" \
    -add "$tmpdir/GLS022_rh.nii.gz" \
    -add "$tmpdir/GLS023_rh.nii.gz" \
    -add "$tmpdir/GLS024_rh.nii.gz" \
    -add "$tmpdir/GLS025_rh.nii.gz" \
    -add "$tmpdir/GLS026_rh.nii.gz" \
    -add "$tmpdir/GLS027_rh.nii.gz" \
    -add "$tmpdir/GLS028_rh.nii.gz" \
    -add "$tmpdir/GLS030_rh.nii.gz" \
    -add "$tmpdir/GLS033_rh.nii.gz" \
    -add "$tmpdir/GLS037_rh.nii.gz" \
    -add "$tmpdir/GLS038_rh.nii.gz" \
    -add "$tmpdir/GLS039_rh.nii.gz" \
    -add "$tmpdir/GLS040_rh.nii.gz" \
    -add "$tmpdir/GLS043_rh.nii.gz" \
    -add "$tmpdir/GLS044_rh.nii.gz" \
    -add "$tmpdir/GLS045_rh.nii.gz" rh.nii.gz
