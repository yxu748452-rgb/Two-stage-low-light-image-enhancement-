import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir) # 创建目录
    imgs = os.listdir(inputdir)
    imgs_target = os.listdir(targetdir)
    t=0
    #print(imgs_target[5])
    for img in imgs:
        groups = ''

        groups += os.path.join(inputdir, img) + '|'
        groups += os.path.join(targetdir,imgs_target[t])
        t=t+1
        # 文件的写操作
        with open(os.path.join(outputdir, 'groups_test_Huawei.txt'), 'a') as f:
            f.write(groups + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='C:/baidunetdiskdownload/Eval/Eval/Huawei/low/', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='C:/baidunetdiskdownload/Eval/Eval/Huawei/high/', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='./data/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.jpg', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()

