import random
import string
import sagemaker
import boto3
import shutil
from joblib import Parallel, delayed
import multiprocessing
from sagemaker import get_execution_role
import os
import cv2
import math
import numpy as np
import pandas  as  pd
import traceback
from skimage import exposure,color, img_as_int, img_as_ubyte
from skimage.io import imread as pngread
from skimage.io import imsave as pngsave
from skimage.morphology import disk
from skimage.filters.rank import autolevel,equalize
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def inputpipeline(files, batchid=randomString(10)):
    def preproc(img):
        selem = disk(60)
        try:
            img = autolevel(img, selem)
            img = exposure.adjust_gamma(img, 2)
            img = cv2.bilateralFilter(img, 9, 75, 75)
        except:
            print(img.shape)
            pass
        return (img)

    def createmultipleinputs(inputpath):
        # pad to square
        im = pngread(inputpath)
        if len(im.shape) == 3:
            print('Images should be grayscale but had dimensions {} - automatically converted'.format(im.shape))
            im = np.sum(im, 2)
        im = np.uint16(img_as_int(exposure.rescale_intensity(im, out_range=(0, 2 ** 15 - 1))))
        imshape = im.shape
        edgediff = np.max(imshape) - np.min(imshape)
        orig = im
        if imshape[1] > imshape[0]:
            orig = cv2.copyMakeBorder(im, math.ceil(edgediff / 2), math.ceil(edgediff / 2), 0, 0, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])
        if imshape[0] > imshape[1]:
            orig = cv2.copyMakeBorder(im, 0, 0, math.ceil(edgediff / 2), math.ceil(edgediff / 2), cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])

        # ==>resize to 1024
        im1024 = cv2.resize(orig, (1024, 1024), interpolation=cv2.INTER_AREA)
        # ==>resize to 720
        im720 = cv2.resize(orig, (720, 720), interpolation=cv2.INTER_AREA)
        # preprocess both
        im1024preproc = preproc(im1024)
        im720preproc = preproc(im720)
        return ([orig, im1024preproc, im720preproc, im1024, im720])

    def populate_inputs(localpaths, batchid=''):
        os.makedirs('/tmp/{}/'.format(batchid), exist_ok=True)
        imlabels = ['orig', 'im1024pp', 'im720pp', 'im1024', 'im720']

        def innerloop(filepath, batchid=batchid, imlabels=imlabels):
            resimages = createmultipleinputs(filepath)
            for idx in range(0, len(resimages)):
                savepath = '/tmp/' + batchid + '/' + batchid + '_' + filepath.split('.')[0].split('/')[-1] + '__' + \
                           imlabels[idx] + '.jpg'
                pngsave(savepath, resimages[idx])

        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(innerloop)(filepath) for filepath in localpaths)
        os.system(
            "aws s3 sync '/tmp/{}/' 's3://sagemaker-eu-west-1-102554356212/submissions/{}/' ".format(batchid, batchid))
        shutil.rmtree('/tmp/{}/'.format(batchid))

    def runbatch(model_id, batchid=''):

        env = {'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600'}
        s3 = boto3.resource('s3')
        s3_resource = boto3.resource('s3')
        sess = sagemaker.Session()
        bucket = sess.default_bucket()
        s3results = s3_resource.Bucket(name=bucket)
        removesamples = [obj.key for obj in s3results.objects.all() if (
                "results_" + model_id in obj.key and batchid in obj.key and (
                "out" in obj.key or "masks" in obj.key))]
        for removeme in removesamples:
            boto3.client('s3').delete_object(Bucket=bucket, Key=removeme)

        transform_job = sagemaker.transformer.Transformer(
            model_name=model_id,
            instance_count=1,
            instance_type='ml.p3.2xlarge',
            strategy='SingleRecord',
            assemble_with='None',
            output_path="s3://{}/results_{}/{}/".format(bucket,model_id, batchid),
            base_transform_job_name='inference-pipelines-batch',
            sagemaker_session=sess,
            accept='image/png',
            env=env)
        transform_job.transform(data='s3://{}/submissions/'.format(bucket),
                                content_type='image/jpeg',
                                split_type=None)
        return (transform_job)

    def merge_multiple_detections(masks):
        """

        :param masks:
        :return:
        """
        IOU_THRESHOLD = 0.6
        OVERLAP_THRESHOLD = 0.8
        MIN_DETECTIONS = 1

        def compute_iou(mask1, mask2):
            """
            Computes Intersection over Union score for two binary masks.
            :param mask1: numpy array
            :param mask2: numpy array
            :return:
            """
            intersection = np.sum((mask1 + mask2) > 1)
            union = np.sum((mask1 + mask2) > 0)
            return intersection / float(union)

        def compute_overlap(mask1, mask2):
            intersection = np.sum((mask1 + mask2) > 1)

            overlap1 = intersection / float(np.sum(mask1))
            overlap2 = intersection / float(np.sum(mask2))
            return overlap1, overlap2

        def sort_mask_by_cells(mask, min_size=50):
            """
            Returns size of each cell.
            :param mask:
            :return:
            """
            cell_num = np.unique(mask)
            cell_sizes = [(cell_id, len(np.where(mask == cell_id)[0]))
                          for cell_id in cell_num if cell_id != 0]

            cell_sizes = [x for x in sorted(
                cell_sizes, key=lambda x: x[1], reverse=True) if x[1 > min_size]]

            return cell_sizes

        cell_counter = 0
        final_mask = np.zeros(masks[0].shape)

        masks_stats = [sort_mask_by_cells(mask) for mask in masks]
        cells_left = sum([len(stats) for stats in masks_stats])

        while cells_left > 0:
            # Choose the biggest cell from available
            cells = [stats[0][1] if len(
                stats) > 0 else 0 for stats in masks_stats]
            reference_mask = cells.index(max(cells))

            reference_cell = masks_stats[reference_mask].pop(0)[0]

            # Prepare binary mask for cell chosen for comparison
            cell_location = np.where(masks[reference_mask] == reference_cell)

            cell_mask = np.zeros(final_mask.shape)
            cell_mask[cell_location] = 1

            masks[reference_mask][cell_location] = 0

            # Mask for storing temporary results
            tmp_mask = np.zeros(final_mask.shape)
            tmp_mask += cell_mask

            for mask_id, mask in enumerate(masks):
                # For each mask left
                if mask_id != reference_mask:
                    # # Find overlapping cells on other masks
                    overlapping_cells = list(np.unique(mask[cell_location]))

                    try:
                        overlapping_cells.remove(0)
                    except ValueError:
                        pass

                    # # If only one overlapping, check IoU and update tmp mask if high
                    if len(overlapping_cells) == 1:
                        overlapping_cell_mask = np.zeros(final_mask.shape)
                        overlapping_cell_mask[np.where(
                            mask == overlapping_cells[0])] = 1

                        iou = compute_iou(cell_mask, overlapping_cell_mask)
                        if iou >= IOU_THRESHOLD:
                            # Add cell to temporary results and remove from stats and mask
                            tmp_mask += overlapping_cell_mask
                            idx = [i for i, cell in enumerate(
                                masks_stats[mask_id]) if cell[0] == overlapping_cells[0]][0]
                            masks_stats[mask_id].pop(idx)
                            mask[np.where(mask == overlapping_cells[0])] = 0

                    # # If more than one overlapping check area overlapping
                    elif len(overlapping_cells) > 1:
                        overlapping_cell_masks = [
                            np.zeros(final_mask.shape) for _ in overlapping_cells]

                        for i, cell_id in enumerate(overlapping_cells):
                            overlapping_cell_masks[i][np.where(
                                mask == cell_id)] = 1

                        for cell_id, overlap_mask in zip(overlapping_cells, overlapping_cell_masks):
                            overlap_score, _ = compute_overlap(
                                overlap_mask, cell_mask)

                            if overlap_score >= OVERLAP_THRESHOLD:
                                tmp_mask += overlap_mask

                                mask[np.where(mask == cell_id)] = 0
                                idx = [i for i, cell in enumerate(masks_stats[mask_id])
                                       if cell[0] == cell_id][0]
                                masks_stats[mask_id].pop(idx)

                    # # If none overlapping do nothing

            if len(np.unique(tmp_mask)) > 1:
                cell_counter += 1
                final_mask[np.where(tmp_mask >= MIN_DETECTIONS)] = cell_counter

            cells_left = sum([len(stats) for stats in masks_stats])

        bin_mask = np.zeros(final_mask.shape)
        bin_mask[np.where(final_mask > 0)] = 255
        return (final_mask)

    def merge_two_masks(maskpaths):
        masks = []
        for mpath in maskpaths:
            binarymask = pngread(mpath)
            distance = ndi.distance_transform_edt(binarymask)
            local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
            markers = ndi.label(local_maxi)[0]
            masks.append(watershed(-distance, markers, mask=binarymask))
        mask = merge_multiple_detections(masks)
        return (mask)

    def merge_masks(modelres, modelids, batchid=''):
        outpath = '/tmp/results/{}/merged/'.format(batchid)
        os.makedirs(outpath, exist_ok=True)

        def savemerge(masklist, outpath=outpath, modelids=modelids):
            mask = np.uint8(merge_two_masks(masklist)) > 0
            savepath = os.path.join(outpath, 'merged_' + masklist[0].split('/')[-1].split(modelids[0])[-1])
            pngsave(savepath, mask)

            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(
                delayed(savemerge)([modelres[0][idx], modelres[1][idx]]) for idx in range(0, len(modelres[0])))

    def batch2masks(model_id, batchid=''):
        s3 = boto3.resource('s3')
        s3_resource = boto3.resource('s3')
        sess = sagemaker.Session()
        bucket = sess.default_bucket()

        s3results = s3_resource.Bucket(name=bucket)
        keys = [obj.key for obj in s3results.objects.all()]
        os.makedirs('/tmp/results/{}'.format(batchid), exist_ok=True)
        savepaths = []
        for s3_object in keys:
            if "results_" + model_id in s3_object and "out" in s3_object:
                s3.meta.client.download_file(bucket, s3_object,
                                             '/tmp/' + s3_object.split('/')[-1])
                with open('/tmp/' + s3_object.split('/')[-1], 'rb') as image:
                    img = image.read()
                    img = bytearray(img)
                    mask = np.array(Image.open(io.BytesIO(img)))
                savepath = '/tmp/results/' + batchid + '/' + model_id + '.'.join(
                    s3_object.split('/')[-1].split('.')[:-1])
                pngsave(savepath, mask)
                os.remove('/tmp/' + s3_object.split('/')[-1])
                savepaths.append(savepath)
        return (savepaths)

    def merge_masks_diff_inputs(groupkeys, batchid=''):
        os.makedirs('/tmp/results/{}/inputmerged/'.format(batchid), exist_ok=True)
        masks = []
        for file in groupkeys:
            binarymask = cv2.resize(pngread(file), (1024, 1024), interpolation=cv2.INTER_AREA)
            distance = ndi.distance_transform_edt(binarymask)
            local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
            markers = ndi.label(local_maxi)[0]
            masks.append(watershed(-distance, markers, mask=binarymask))
            try:
                binarymask = merge_two_masks(outpaths)
                distance = ndi.distance_transform_edt(binarymask)
                local_maxi = peak_local_max(distance, labels=binarymask, footprint=np.ones((3, 3)), indices=False)
                markers = ndi.label(local_maxi)[0]
            except:
                mask = watershed(-distance, markers, mask=binarymask)
                pass
            savepath = os.path.join('/tmp/results/' + batchid + '/inputmerged/',
                                    file.split('/')[-1].split('__')[0].replace('merged_', 'inputmerged_') + '.jpg')
            pngsave(savepath, np.uint8(mask > 0))
    try:
        modelids = ["fresh-train-trial-2019-07-28-08-49-49-994", "semantic-segmentatio-190726-1931-032-e7d26e04"]
#         files = os.listdir(inputpath)
        print(files)
        [os.rename(f,f.replace('_', '-')) for f in files]
        files = [f.replace('_','-') for f in files] #we use __ to split multiple preprocessings so must remove them from file name
        files = [f for f in files if '.jpg' in f or '.png' in f or '.tif' in f]
        populate_inputs(files, batchid=batchid)
        tj = [runbatch(model) for model in modelids]
        tj[len(tj)].wait()
        results = [batch2masks(mid, batchid=batchid) for mid in modelids]
        twomods = list(
            set([r.split(modelids[0])[-1] for r in results[0]]) & set([r.split(modelids[1])[-1] for r in results[1]]))
        firstfiles = [r for r in results[0] if r.split(modelids[0])[-1] in twomods]
        secondfiles = [r for r in results[1] if r.split(modelids[1])[-1] in twomods]
        merge_masks((firstfiles, secondfiles), modelids = modelids, batchid=batchid)
        keys = []
        for root, dirnames, filenames in os.walk('/tmp/results/{}/merged/'.format(batchid)):
            for files in filenames:
                if ('.jpg' in files or '.png' in files or '.tif' in files):
                    keys.append(os.path.join('/tmp/results/{}/merged/'.format(batchid), files))
        df = pd.DataFrame({'keys': keys, 'orig_name': [k.split('/')[-1].split('__')[0].split('.jpg')[0] for k in
                                                       keys]})  # add change file names if they have __
        originals = np.unique(df['orig_name'].values)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(
            delayed(merge_masks_diff_inputs)(df['keys'].loc[df['orig_name'] == org].values, batchid) for org in originals)
        os.system(
            "aws s3 sync '/tmp/results/{}/inputmerged/' 's3://sagemaker-eu-west-1-102554356212/results_merged/{}/' ".format(
                batchid, batchid))
        # write_email("send_link", submit_email = submitter)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
#         print("error: {}".format(str(e)))
        # write_email("error_report", errormessage = str(e))

