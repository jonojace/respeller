#INDIR='/home/s1785140/data/ljspeech_fastpitch/wavs_and_labs_test'
#OUTDIR='/home/s1785140/data/ljspeech_fastpitch/aligns_test'

INDIR='/home/s1785140/data/ljspeech_fastpitch/wavs_and_labs'
OUTDIR='/home/s1785140/data/ljspeech_fastpitch/aligns'

rm $OUTDIR/*

ALIGN_OUTDIR='/home/s1785140/Documents/MFA/wavs_and_labs_temp_pretrained_aligner/pretrained_aligner/textgrids'
TMP='/home/s1785140/data/ljspeech_fastpitch/wavs_and_labs_temp'
mkdir -p $TMP

rm $OUTDIR/*
rm $ALIGN_OUTDIR/*
rm $TMP/*

n=500 # how many files to put into temp folder at a time
# ensure this is even! as we must have a lab for each wav

# get all wavs and labs in directory
cd $INDIR
files=(*)

for ((i=0; i < ${#files[@]}; i+=n)); do
        echo Aligning "${files[@]:i:n}" ...

        # copy wavs and labs into TMP
        cp "${files[@]:i:n}" $TMP

        # run mfa alignment
        mfa align --clean $TMP english_us_arpa english_us_arpa $OUTDIR
        cp $ALIGN_OUTDIR/* $OUTDIR

        # remove wavs and labs from TMP
        rm $ALIGN_OUTDIR/*
        rm $TMP/*
done
