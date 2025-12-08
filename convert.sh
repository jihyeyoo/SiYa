OUTPUT_DIR="../"   
mkdir -p $OUTPUT_DIR

for ID in TENX24 TENX39 TENX97 MISC61 TENX153; do
    mkdir -p ${OUTPUT_DIR}/${ID}/metadata \
             ${OUTPUT_DIR}/${ID}/patches \
             ${OUTPUT_DIR}/${ID}/st

    cp metadata/${ID}.json      ${OUTPUT_DIR}/${ID}/metadata/metadata.json
    cp patches/${ID}.h5         ${OUTPUT_DIR}/${ID}/patches/patches.h5
    cp st/${ID}.h5ad            ${OUTPUT_DIR}/${ID}/st/st.h5ad
done
