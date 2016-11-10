# checks for a free gpu or waits till one is available
source /data/ASR1/tools/theanoenv/setup
export PATH=/data/ASR1/tools/python27/bin:$PATH
export PYTHONPATH=/opt/python27/lib/python2.7/site-packages:/data/ASR1/tools/theanoenv/lib/python2.7/site-packages:$PYTHONPATH

if [ -n "$PBS_JOBID" ]; then
    # get slot we occupy
    gpu=`qstat -n $PBS_JOBID|awk ' END { split ($NF, a, "/"); printf ("gpu%s\n", a[2]) } '`
    echo "Using $gpu (from PBS_JOBID)"
else

    while true; do
        i=`nvidia-smi|awk '/Compute processes/,0 {print $2}'|tail -n +4|head -n -1|sort -g|awk '{array[int($1)]=1} END{for(x=0;x<=7;x++) if(array[x]=="") {print x; break}}'`
        if [ -n "$i" ]; then 
            gpu=gpu$i;
            echo "using $gpu !"
            break;
        else
            echo "No free GPUs! sleeping for 5 minutes"
            sleep 300
        fi
    done
fi

export THEANO_FLAGS="cuda.root=/usr/local/cuda,device=$gpu,floatX=float32,config.nvcc.fastmath=True,allow_gc=False"

pdnndir=/data/ASR5/babel/ymiao/tools/pdnn    # set according to your own directory
export PYTHONPATH=$pdnndir:$PYTHONPATH

python $pdnndir/cmds/run_DNN.py --train-data "train.pickle.gz" \
                                --valid-data "valid.pickle.gz" \
                                --nnet-spec "784:1024:1024:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:200" --model-save-step 20 \
                                --param-output-file dnn.param --cfg-output-file dnn.cfg >& dnn.training.log

