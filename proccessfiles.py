
class processfiles:
    

    def proccessfigure8files(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
        train_channel = prefix + '/train'
        validation_channel = prefix + '/validation'
        train_annotation_channel = prefix + '/train_annotation'
        validation_annotation_channel = prefix + '/validation_annotation'
        if 'tif' in f:
            file = f
            if 'segproj/training_data/train/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/raw.tif')
                inverted_img = util.invert(imread('/tmp/raw.tif'))
                jpgpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.jpg'
                num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
                image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
                imsave(jpgpath,img_as_ubyte(image))
                sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=train_channel)
            elif 'segproj/training_data/val/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/raw.tif')
                inverted_img = util.invert(imread('/tmp/raw.tif'))
                jpgpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.jpg'
                num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
                image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
                imsave(jpgpath,img_as_ubyte(image))
                sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=validation_channel)
        elif 'instances_ids.png' in f:
            file = f
            if 'segproj/training_data/train/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/instances_ids.png')
                pngpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.png'
                im1 = pngread('/tmp/instances_ids.png')
                num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
                image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
                image = img_as_ubyte(image)
                im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
                im2 = img_as_int(im)
                im2 = np.uint8((im2>0))
                im2 =  cv2.resize(im2, (1024,1024), interpolation = cv2.INTER_AREA)            
                pngsave(pngpath,im2, check_contrast=False)
                sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
            elif 'segproj/training_data/val/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/instances_ids.png')
                pngpath = '/tmp/'+'fig8_'+file.split('/')[-2]+'_'+'raw.png'
                im1 = pngread('/tmp/instances_ids.png')
                num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
                image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
                image = img_as_ubyte(image)
                im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
                im2 = img_as_int(im)
                im2 = np.uint8((im2>0))
                im2 =  cv2.resize(im2, (1024,1024), interpolation = cv2.INTER_AREA)            
                pngsave(pngpath,im2, check_contrast=False)
                sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)


    def proccessusiigacifiles(f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):
        train_channel = prefix + '/train'
        validation_channel = prefix + '/validation'
        train_annotation_channel = prefix + '/train_annotation'
        validation_annotation_channel = prefix + '/validation_annotation'
        if 'tif' in f:
            ranNUM = random.randint(0,1000000)
            file = f
            if 'segproj/usiigaci_train_data/train/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/raw'+str(ranNUM)+'.tif')
                inverted_img = util.invert(imread('/tmp/raw'+str(ranNUM)+'.tif'))
                jpgpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.jpg'
                num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
                image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
                image =  cv2.resize(img_as_ubyte(image), (1024,1024), interpolation = cv2.INTER_AREA)
                image,_ = rolling_ball_filter(image, ball_radius = 20, spacing = 1, top=False)
                imsave(jpgpath,image)
                sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=train_channel)
            elif 'segproj/usiigaci_train_data/val/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/raw'+str(ranNUM)+'.tif')
                inverted_img = util.invert(imread('/tmp/raw'+str(ranNUM)+'.tif'))
                jpgpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.jpg'
                num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
                image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
                image =  cv2.resize(img_as_ubyte(image), (1024,1024), interpolation = cv2.INTER_AREA)
                image,_ = rolling_ball_filter(image, ball_radius = 20, spacing = 1, top=False)
                imsave(jpgpath,image)
                sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=validation_channel)
        elif 'instances_ids.png' in f:
            ranNUM = random.randint(0,1000000)
            file = f
            if 'segproj/usiigaci_train_data/train/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/instances_ids'+str(ranNUM)+'.png')
                pngpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.png'
                im1 = pngread('/tmp/instances_ids'+str(ranNUM)+'.png')
                num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
                image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
                image = img_as_ubyte(image)
                im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
                im2 = img_as_int(im)
                im2 = np.uint8((im2>0))
                im2 =  cv2.resize(im2, (1024,1024), interpolation = cv2.INTER_AREA)            
                pngsave(pngpath,im2, check_contrast=False)
                sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
            elif 'segproj/usiigaci_train_data/val/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/instances_ids'+str(ranNUM)+'.png')
                pngpath = '/tmp/'+'usiigaci_'+file.split('/')[-2]+'_'+'raw.png'
                im1 = pngread('/tmp/instances_ids'+str(ranNUM)+'.png')
                num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
                image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
                image = img_as_ubyte(image)
                im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
                im2 = img_as_int(im)
                im2 = np.uint8((im2>0))
                im2 =  cv2.resize(im2, (1024,1024), interpolation = cv2.INTER_AREA)            
                pngsave(pngpath,im2, check_contrast=False)
                sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)

    def proccessliorfiles(self, f, s3 = s3, sess = sess, bucket = bucket, prefix = prefix):

        if 'tif' in f:
            ranNUM = random.randint(0,1000000)
            file = f
            file = file.replace('._','')
            if 'segproj/liorp_training_data/train/' in f:
                self.s3.meta.client.download_file('meadata', file, '/tmp/raw'+str(ranNUM)+'.tif')
                image = imread('/tmp/raw'+str(ranNUM)+'.tif')
                image = clahe.apply(image)
                image =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)
                inverted_img,_ = rolling_ball_filter(image, ball_radius = 20, spacing = 1, top=False)
                jpgpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.jpg'
                num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
                image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
                imsave(jpgpath,img_as_ubyte(image))
                sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=train_channel)
            elif 'segproj/liorp_training_data/val/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/raw'+str(ranNUM)+'.tif')
                image = imread('/tmp/raw'+str(ranNUM)+'.tif')
                image = clahe.apply(image)
                image =  cv2.resize(image, (1024,1024), interpolation = cv2.INTER_AREA)
                inverted_img,_ = rolling_ball_filter(image, ball_radius = 20, spacing = 1, top=False)
                jpgpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.jpg'
                num = int(''.join(filter(str.isdigit, str(inverted_img.dtype)))) - 1
                image = exposure.rescale_intensity(inverted_img, out_range=(0, 2**num - 1))
                imsave(jpgpath,img_as_ubyte(image))
                sess.upload_data(path=jpgpath, bucket=bucket, key_prefix=validation_channel)
        elif 'instances_ids.png' in f:
            ranNUM = random.randint(0,1000000)
            file = f
            file = file.replace('._','')
            if 'segproj/liorp_training_data/train/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/instances_ids'+str(ranNUM)+'.png')
                pngpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.png'
                im1 = pngread('/tmp/instances_ids'+str(ranNUM)+'.png')
                num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
                image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
                image = img_as_ubyte(image)
                im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
                im2 = img_as_int(im)
                im2 = np.uint8((im2>0))
                im2 =  cv2.resize(im2, (1024,1024), interpolation = cv2.INTER_AREA)            
                pngsave(pngpath,im2, check_contrast=False)
                sess.upload_data(path=pngpath, bucket=bucket, key_prefix=train_annotation_channel)
            elif 'segproj/liorp_training_data/val/' in f:
                s3.meta.client.download_file('meadata', file, '/tmp/instances_ids'+str(ranNUM)+'.png')
                pngpath = '/tmp/'+'liorp_'+file.split('/')[-2]+'_'+'raw.png'
                im1 = pngread('/tmp/instances_ids'+str(ranNUM)+'.png')
                num = int(''.join(filter(str.isdigit, str(im1.dtype)))) - 1
                image = exposure.rescale_intensity(im1, out_range=(0, 2**num - 1))
                image = img_as_ubyte(image)
                im = mark_boundaries(image, im1, color = [0,0,0], outline_color = [0,0,0], mode='outer', background_label=0)
                im2 = img_as_int(im)
                im2 = np.uint8((im2>0))
                im2 =  cv2.resize(im2, (1024,1024), interpolation = cv2.INTER_AREA)            
                pngsave(pngpath,im2, check_contrast=False)
                sess.upload_data(path=pngpath, bucket=bucket, key_prefix=validation_annotation_channel)
                
        def _init_(self, s3, sess, bucket, prefix)
                self.train_channel = prefix + '/train'
                self.validation_channel = prefix + '/validation'
                self.train_annotation_channel = prefix + '/train_annotation'
                self.validation_annotation_channel = prefix + '/validation_annotation'
                self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))