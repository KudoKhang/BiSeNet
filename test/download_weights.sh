mkdir checkpoints
gdown 1igAhgFWELyo5YLGQLRe_Q5PuA4CxyPhn -O checkpoints/best_model_CeFiLaIb.pth

mkdir dataset
cd dataset
gdown 14OZ35B6baZ9HO5Bh_NsKDNVxlN-mJLrM
unzip -qq Figaro_1k.zip
rm -rf Figaro_1k.zip
rm -rf __MACOSX
cd ..
echo "Download weight and datasets successfully!"