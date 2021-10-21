import glob
import os
import random
import shutil


def copy_images(src_image_paths, dist_dir):
    os.makedirs(dist_dir)
    print(f'Make {dist_dir} directory.')
    for src_image_path in src_image_paths:
        dist_image_path = os.path.join(dist_dir, os.path.basename(src_image_path))
        shutil.copyfile(src_image_path, dist_image_path)
        print(f'Copy {src_image_path} to {dist_image_path}.')
    return


def main():
    names = ['work', 'not_work']

    for name in names:
        images_dir = f'./data/{name}'
        print(images_dir)
        image_paths = glob.glob(f'{images_dir}/*.jpg')
        random.shuffle(image_paths)
        n_images = len(image_paths)
        print(f'n_images => {n_images}')
        assert n_images > 50

        n_test = 50
        n_train = int((n_images - n_test) * 0.7)
        n_val = int((n_images - n_test) * 0.3)
        print(f'n_train, n_val, n_test = {n_train}, {n_val}, {n_test}')
        assert n_images == n_train + n_val + n_test

        train_paths = image_paths[:n_train]
        val_paths = image_paths[n_train:n_train + n_val]
        test_paths = image_paths[n_train + n_val:]

        copy_images(train_paths, f'./dataset/train/{name}')
        copy_images(val_paths, f'./dataset/validation/{name}')
        copy_images(test_paths, f'./dataset/test/{name}')
    return


if __name__ == "__main__":
    main()
