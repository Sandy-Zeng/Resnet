import os

rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['cifar-10']
depths = [20]
gpu_id = '0'
batchsize = 128
epoch = 200
exp_dir = './tb_dir/cifar_exp/test_128bs_200epoch'
res = exp_dir + 'res.txt'

for data in datasets:
    for depth in depths:
        cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
                                %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        # cmd = 'python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #       % (data, depth, res, gpu_id, batchsize, epoch, exp_dir)
        os.system(cmd)



