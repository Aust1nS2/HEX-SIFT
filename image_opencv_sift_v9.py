import numpy as np
import cv2
import tifffile
from optparse import OptionParser
from ome_types import from_tiff, from_xml, to_xml, model
from ome_types.model.simple_types import UnitsLength

parser = OptionParser()
parser.add_option("-i", "--image",
                  action="store", type="string", dest="he_raw_path",
                  help="path to the raw input H&E image to be aligned to dapi image")
parser.add_option("-d", "--dapi",
                  action="store", type="string", dest="dapi_path",
                  help="path to the input dapi image from xenium platform (use the morphology_mip.ome.tif file)")
parser.add_option("-b", "--blue",
                  action="store_true", dest="blue", default=False,
                  help="only use the blue channel from the H&E image for image alignment")
parser.add_option("-f", "--flip",
                  action="store_true", dest="flip", default=False,
                  help="flips the image before aligning it. The SIFT alignment algorithm this script utilized can handle image rotation, translation (move left/right or up/down), and scaling (uniformly stretching and shrinking). However, it cannot flip an image. If the image needs to be flipped in order to align the slide then use this option.")
parser.add_option("-s", "--scaling",
                  action="store", type="int", dest="downscaling", default=4,
                  help="Specify a whole number from 0-7. Default is 4. It is not recommended that you change this value. This is the level of dapi image downscaling that will be used to calculate the Homograpphy matrix for image alignment. A level of 4 means that the Y and X axes of the H&E image will be divided by 2^4 during initial homography matrix calculation. The Homography matrix will then be scaled back up by the resulting scale matrix for use in the full size H&E alignment")
parser.add_option("-m", "--matches_limit",
                  action="store", type="int", dest="matches_limit", default=20,
                  help="Specify any whole number less than the total number of keypoint matches found. Default is 20. It is not recommended that you change this value for an initial alignment. Keypoints and their descriptors of an image are identified using the SIFT algorithm and they are then matched using a FLANN matcher. The top n keypoint matches are then used for image alignment. If there are a large number of potentially good keypoint matches then increasing this number can improve the alignment. Otherwise decreasing it can hurt the alignment.")

(options, args) = parser.parse_args()

#he_raw_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/output-XETG00122__0011117__HT434P1-S1H2Fp1Us1_1__20230919__220650/opencv_sift_flann/0011117_Scan1_crop_HT434P1-S1H2Fp1Us1_1.ome.tif"
with tifffile.TiffFile(options.he_raw_path) as tif:
    he_image = tif.series[0].levels[0].asarray()

#dapi_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/output-XETG00122__0011117__HT434P1-S1H2Fp1Us1_1__20230919__220650/opencv_sift_flann/morphology_mip.ome.tif"
with tifffile.TiffFile(options.dapi_path) as tif:
    dapi_image = tif.series[0].levels[options.downscaling].asarray()

def writing_ome_tif(FRAMES = np.zeros((2,512,512)),
                     subresolutions = 8,
                     dapi_image_level=0,
                     outfile_name = "test.ome.tif",
                     isRGB=False,
                     dtype='uint8',
                     channel_names=[],
                     make_thmubnail=False):
                    
    dapi_image_dict = {0:0.2125,
                       1:0.4250,
                       2:0.8500,
                       3:1.7000,
                       4:3.4000,
                       5:6.8000,
                       6:13.6000,
                       7:27.2000}
    print("Writing",outfile_name,"file")
    if isRGB:
        colormode='rgb'
    else:
        colormode='minisblack'
    FRAMES = np.asarray(FRAMES)
    if dtype=='uint8':
        significant_bits=8
        numpy_dtype=np.uint8
        # rescale the image using cv2.normalize (see https://stackoverflow.com/questions/24444334/numpy-convert-8-bit-to-16-32-bit-image and  https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd)
        if type(np.max(FRAMES)) != numpy_dtype:
            FRAMES = (FRAMES/65535)*255
        FRAMES = np.asarray(FRAMES).astype(numpy_dtype)
        print(np.max(FRAMES))
        print(type(FRAMES[0,0,0]))
        print(type(np.max(FRAMES)))
    else:
        significant_bits=16
        dtype='uint16'
        numpy_dtype=np.uint16
        if type(np.max(FRAMES)) != numpy_dtype:
            FRAMES = np.asarray(FRAMES).astype(numpy_dtype)
            FRAMES = (FRAMES/255)*65535
        FRAMES = np.asarray(FRAMES).astype(numpy_dtype)
        print(np.max(FRAMES))
        print(type(FRAMES[0,0,0]))
        print(type(np.max(FRAMES)))
    FRAMES = FRAMES[np.newaxis, ...] #adding the Z stack level
    FRAMES = FRAMES[np.newaxis, ...] #adding the time level
    print(FRAMES.shape) #(2, 48340, 64750)
    subresolutions = 8 #2
    pixelsize = dapi_image_dict[dapi_image_level]  # micrometer
    #####
    #data = numpy.random.randint(0, 255, (8, 2, 512, 512, 3), 'uint8')
    # this function is writtent to handle arrays with that look like (8, 2, 3, 512, 512)
    # this function is writtent to handle arrays with that look like (8, 2, 512, 512, 3)
    # figuring out if the channels are in the 3rd axis of the array or the 5th
    # create a list of numpy axis (dimension) lengths
    dim_len = list(FRAMES.shape)
    #print(dim_len)
    # loop over the dimension lengths and a tuple of their order in the FRAMES.shape creating a list of tuples in list comprehension 
    # and then sort it by the value of the first element in the tuple (the dimension length)
    dim_len_sorted = sorted([(i, j) for i, j in zip(dim_len, range(len(dim_len)))], key=lambda tup: tup[0], reverse=True)
    #print(dim_len_sorted)
    # take the top 2 j from the above list as these are the axes of the FRAME array that store the X and Y axis information.
    biggest_FRAME_indices = (dim_len_sorted[0][1], dim_len_sorted[1][1])
    #print(biggest_FRAME_indices)
    # check if the axis 3 is in the tuple of biggest_FRAME_indices
    if 2 in biggest_FRAME_indices:
        # if it is then axis 2 and 3 are the y and x of the array respectively otherwise it is axis 3 and 4
        index_list = (4, 2, 3) # TZYXC <- the case for H&E images TZYXC so we will index 4, then 2, then 3 for channel, y, x
        FRAMES_new = np.zeros((FRAMES.shape[index_list[0]], FRAMES.shape[index_list[1]], FRAMES.shape[index_list[2]]), dtype=numpy_dtype)
        FRAMES_new = FRAMES_new[np.newaxis, ...]
        FRAMES_new = FRAMES_new[np.newaxis, ...]
        #print(FRAMES_new.shape)
        for i in range(FRAMES.shape[index_list[0]]):
            channel = FRAMES[0,0,:,:,i]
            #print(channel.shape)
            FRAMES_new[0,0,i,:,:] = channel
        FRAMES = FRAMES_new
        index_list = (2, 3, 4)
    else:
        index_list = (2, 3, 4) # TZCYX <- the case for multiple stacked monocrome image so we will index 2, then 3, then 4 for channel, y, x
    #####
    with tifffile.TiffWriter(outfile_name, ome=True, bigtiff=True) as tif:
        #thumbnail = (FRAMES[0, 0, 1, ::256, ::256] >> 2).astype('uint8')
        #tif.write(thumbnail, metadata={'Name': 'thumbnail'})
        metadata={
            # 'axes': 'YXS',
            'axes': 'TZCYX',
            'SignificantBits': significant_bits, #10
            'TimeIncrement': 0.1,
            'TimeIncrementUnit': 's',
            'PhysicalSizeX': pixelsize,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': pixelsize,
            'PhysicalSizeYUnit': 'µm',
            # currently the metadata channel and plane information is not accurately updated based on the kind of image. will probably need to input a dictionary as a keyword argument and then use that as the input here.
            'Channel': {'Name': channel_names},
            'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
        }
        ########
        #o = model.OME()
        #o.images.append(
        #    model.Image(
        #        id='Image:0',
        #        pixels=model.Pixels(
        #            dimension_order='XYCZT',
        #            size_c=FRAMES.shape[2],
        #            size_t=FRAMES.shape[0],
        #            size_x=FRAMES.shape[4],
        #            size_y=FRAMES.shape[3],
        #            size_z=FRAMES.shape[1],
        #            type=dtype,
        #            big_endian=False,
        #            channels=[model.Channel(id=f'Channel:{i}', name=c) for i, c in enumerate(channel_names)],
        #            physical_size_x=1 / pixelsize,
        #            physical_size_y=1 / pixelsize,
        #            physical_size_x_unit='µm',
        #            physical_size_y_unit='µm'
        #        )
        #    )
        #)
        #im = o.images[0]
        #for i in range(len(im.pixels.channels)):
        #    im.pixels.planes.append(model.Plane(the_c=i, the_t=0, the_z=0))
        #im.pixels.tiff_data_blocks.append(model.TiffData(plane_count=len(im.pixels.channels)))
        ########

        options = dict(
            photometric=colormode,
            tile=(1024, 1024), #tile=(128, 128),
            compression='zlib', #compression='jpeg',
            compressionargs={'level': 8}, #worked when this was 4
            resolutionunit='CENTIMETER',
                #resolutionunit='MICROMETER',
                #resolution=pixelsize,
                #imageJ=True,
            maxworkers=8 #2 #This is the number of threads to use when saving
        )
        if subresolutions > 0:
            print("writing full resolution image")
            tif.write(
                FRAMES,
                subifds=subresolutions,
                resolution=(1e4 / pixelsize, 1e4 / pixelsize),
                metadata=metadata,
                #imagej=True, #this is a flag of tifffile.imwrite() not tifffile.TiffWriter.write()
                **options
            )
        # write pyramid levels to the two subifds
        # in production use resampling to generate sub-resolution images
            print("writing subresolutions")
            for level in range(subresolutions):
                mag = 2**(level + 1)
                print("shinking image by", mag)
                #print("channels first")
                FRAMES_small = np.zeros((FRAMES.shape[index_list[0]], FRAMES.shape[index_list[1]]//mag, FRAMES.shape[index_list[2]]//mag), dtype=numpy_dtype)
                #print(FRAMES_small.shape)
                for idx in range(FRAMES.shape[index_list[0]]):
                    img_layer = FRAMES[0,0,idx, :, :]
                    #print(img_layer.shape)
                    img_layer_small = cv2.resize(img_layer, (img_layer.shape[1]//mag, img_layer.shape[0]//mag), interpolation=cv2.INTER_AREA)
                    #print(img_layer_small.shape)
                    FRAMES_small[idx, :, :] = img_layer_small
                FRAMES_small = FRAMES_small[np.newaxis, ...]
                FRAMES_small = FRAMES_small[np.newaxis, ...]
                print(FRAMES_small.shape)
                print(type(FRAMES_small[0,0,0,0,0]))
                tif.write(
                    FRAMES_small,
                    #FRAMES[..., ::mag, ::mag],
                    subfiletype=1,
                    resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
                    #imagej=True, #this is a flag of tifffile.imwrite() not tifffile.TiffWriter.write()
                    **options
                )
        else:
            print("writing image with no subresolutions")
            tif.write(
                FRAMES,
                #subifds=subresolutions,
                resolution=(1e4 / pixelsize, 1e4 / pixelsize),
                metadata=metadata,
                #imagej=True, #this is a flag of tifffile.imwrite() not tifffile.TiffWriter.write()
                **options
            )
        # add a thumbnail image as a separate series
        # it is recognized by QuPath as an associated image
        if make_thmubnail==True:
            img_layer = FRAMES[0,0,0, :, :]
            if img_layer.shape[0] > 511 and img_layer.shape[1] > 511:
                img_thumbnail = cv2.resize(img_layer, (img_layer.shape[1]//(2**8), img_layer.shape[0]//(2**8)), interpolation = cv2.INTER_AREA)
                print(img_thumbnail.shape)
                img_thumbnail = img_thumbnail[np.newaxis, ...]
                img_thumbnail = img_thumbnail[np.newaxis, ...].astype(numpy_dtype)
                #thumbnail = (FRAMES[0, 0, 1, ::256, ::256] >> 2).astype('uint8')
                tif.write(img_thumbnail, metadata={'Name': 'thumbnail'})
        ########
        #xml_str = to_xml(o)
        #tif.overwrite_description(xml_str.encode())
        ########

# Check to make sure that the image is not taller or wider than the sint16 (signed 16-bit integer limit) 
# if the image is larger then we need to resize it before starting or the final file alignment will fail.
if he_image.shape[0] > 32766:
    print("image height is larger than the signed 16-bit integer limit. The image needs to be re-scaled prior to alignment. Output aligned image will be a sligtly lower resolution image as a result.")
    scale = he_image.shape[0]/32766
    he_image = cv2.resize(he_image, (he_image.shape[1]//scale, he_image.shape[0]//scale), interpolation = cv2.INTER_AREA)
elif he_image.shape[1] > 32766:
    print("image width is larger than the signed 16-bit integer limit. The image needs to be re-scaled prior to alignment. Output aligned image will be a sligtly lower resolution image as a result.")
    scale = he_image.shape[1]/32766
    he_image = cv2.resize(he_image, (int(he_image.shape[1]//scale), int(he_image.shape[0]//scale)), interpolation = cv2.INTER_AREA)

# if the image needs
if options.flip:
    print("H&E image has been flipped over y-axis")
    he_image_flip = cv2.flip(he_image,0)
else:
    he_image_flip = he_image

# When I look at the grayscale image there are clearly regions that are almost entirely stroma that are a deep pink in the HE image and more gray on the grayscaled image.
# In HE images the nuclei are typically purple/blue, while the stroma are typically a stronger pink or white. So instead of grayscaling I might try building in an option to only use the blue channel instead of the full RGB
he_blue = he_image_flip[:,:,0]
he_green = he_image_flip[:,:,1]
he_red = he_image_flip[:,:,2]
cv2.imwrite("he_blue.tif", he_blue)
cv2.imwrite("he_green.tif", he_green)
cv2.imwrite("he_red.tif", he_red)

if options.blue:
    he_image_gray = he_image_flip[:,:,0]
else:
    he_image_gray = cv2.cvtColor(he_image_flip, cv2.COLOR_BGR2GRAY)


he_resize_scale = int(2**options.downscaling)
#dapi_image = cv2.resize(dapi_image, (dapi_image.shape[1]//he_resize_scale, dapi_image.shape[0]//he_resize_scale))
dapi_image_scale_0_1 = (dapi_image - np.min(dapi_image))/(np.max(dapi_image)-np.min(dapi_image))
dapi_image_scale_0_255 = dapi_image_scale_0_1*255
dapi_image_scale_0_255 = dapi_image_scale_0_255.astype(np.uint8)
dapi_image_inverted_small = 255 - dapi_image_scale_0_255


print(dapi_image_inverted_small.shape)

#dapi_image_inverted_small = cv2.resize(dapi_image_inverted_small, (512, 512))
#he_image_gray = cv2.resize(he_image_gray, (512, 512))

he_image_gray_small = cv2.resize(he_image_gray, (he_image_gray.shape[1]//he_resize_scale, he_image_gray.shape[0]//he_resize_scale), interpolation = cv2.INTER_AREA)
print(he_image_gray_small.shape)
cv2.imwrite("he_small.tif", he_image_gray_small)

he_gray_small_blur = cv2.GaussianBlur(he_image_gray_small, (3,3), 0)
he_gray_small_blur_thresh = cv2.threshold(he_gray_small_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(he_gray_small_blur_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 150: #originally was 5500
        cv2.drawContours(he_gray_small_blur_thresh, [c], -1, (0,0,0), -1)

# Morph close and invert image
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)) #André said to try disk size 9,9
#close = 255 - cv2.morphologyEx(he_gray_small_blur_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
he_close = 255 - cv2.morphologyEx(he_gray_small_blur_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
he_close_erosion = cv2.erode(he_close, kernel, iterations = 2)


# tried 1 itteration but there was some regions of adipose tissue or ducts that were not included that probably should be inlcuded so I used 2. 
# To fill in even more either change the number of iterations to 2 or change
# the structuring element shape from 9x9 to something else.
cv2.imwrite('he_thresh.tif', he_gray_small_blur_thresh)
cv2.imwrite('he_close.tif', he_close)
cv2.imwrite('he_erode_2.tif', he_close_erosion)


he_mask = cv2.bitwise_not(he_close_erosion)
he_image_gray_small_invert = cv2.bitwise_not(he_image_gray_small)

he_image_gray_small_invert_filter =  cv2.bitwise_and(he_image_gray_small_invert, he_image_gray_small_invert, mask=he_mask)
he_image_gray_small_filter = cv2.bitwise_not(he_image_gray_small_invert_filter)

cv2.imwrite('he_masked.tif', he_image_gray_small_filter)

print("Finding keypoints and descriptors")
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(dapi_image_inverted_small, None)
kp2, des2 = sift.detectAndCompute(he_image_gray_small_filter, None)

#keypoint_list_to_dump = []
#for i in range(len(kp1)):
#    point = kp1[i]
#    desc = des1[i]
#    temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, desc)
#    keypoint_list_to_dump.append([temp])

import pickle
#with open(r"keypoint_list_to_dump.pickle", "wb") as output_file:
#    pickle.dump(keypoint_list_to_dump, output_file)

print("Matching keypoints")
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50) # or pass empty dictionary
#flann = cv2.FlannBasedMatcher(index_params,search_params)
flann = cv2.FlannBasedMatcher(index_params, {})
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
#matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
#for i,(m,n) in enumerate(matches):
#    if m.distance < 0.65*n.distance:
#        matchesMask[i]=[1,0]
        
#draw_params = dict(matchColor = (0,255,0),
#    singlePointColor = (255,0,0),
#    matchesMask = matchesMask,
#    flags = cv2.DrawMatchesFlags_DEFAULT)

# Draw matches
#img3 = cv2.drawMatchesKnn(dapi_image_inverted_small,kp1,he_image_gray_small_filter,kp2,matches,None,**draw_params)
#cv2.imwrite('FLANN_matches.tif', img3)

test_sort = sorted([(i[0].distance/i[1].distance, j) for i, j in zip(matches, range(len(matches)))], key=lambda tup: tup[0])
retain_good = options.matches_limit
good_matches = []
for i in range(retain_good):
    match_index = test_sort[i][1]
    good_matches.append(matches[match_index][0])

print("Number of good matches rank:", len(good_matches))

good = [m1 for (m1, m2) in matches if m1.distance < 0.65 * m2.distance]

print("Number of good matches: %s"%len(good))

good = good_matches

print("Aligning_small")
canvas = he_image_gray_small_filter.copy()
MIN_MATCH_COUNT = 4
## (7) find homography matrix
## When there are enough robust matching point pairs (at least 4)
if len(good)>MIN_MATCH_COUNT:
    ## Extract corresponding point pairs from matches
    ## queryIndex for the small object, trainIndex for the scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #print(src_pts)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #print(dst_pts)
    ## find homography matrix in cv2.RANSAC using good match points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 5.0)
    #print(M)
    ##  Mask, used to draw the point pairs used in calculating the homography matrix
    #matchesMask2 = mask.ravel().tolist()
    ## Calculate the distortion in Figure 1, which is the corresponding position in Figure 2.
    h,w = dapi_image_inverted_small.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #print(pts)
    dst = cv2.perspectiveTransform(pts,M)
    ## Draw border
    cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))


## (8) drawMatches
matched = cv2.drawMatches(dapi_image_inverted_small,kp1,canvas,kp2,good,None)#,**draw_params)

## (9) Crop the matched region from scene
h,w = dapi_image_inverted_small.shape[:2]
#print("height:",h,", width:",w)
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#print("pts",pts)
dst = cv2.perspectiveTransform(pts,M)
#print("dst",dst)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
#print("persepectiveM", perspectiveM)
found = cv2.warpPerspective(he_image_gray_small,perspectiveM,(w,h))

## (10) save and display
cv2.imwrite("matched.tif", matched)
cv2.imwrite("found.tif", found)
print("Small image aligned")
print("Aligning large image")

# Read in the full size dapi image
with tifffile.TiffFile(options.dapi_path) as tif:
    dapi_image = tif.series[0].levels[0].asarray()
# re-scale the dapi image uint8 instead of unit16
dapi_image_scale_0_1 = (dapi_image - np.min(dapi_image))/(np.max(dapi_image)-np.min(dapi_image))
dapi_image_scale_0_255 = dapi_image_scale_0_1*255
dapi_image_scale_0_255 = dapi_image_scale_0_255.astype(np.uint8)
dapi_image_inverted = 255 - dapi_image_scale_0_255

print(dapi_image_inverted.shape)
print(he_image_flip.shape)
print(he_image_gray.shape)

canvas_large = he_image_gray.copy()
h,w = dapi_image_inverted.shape[:2]

s_y = he_image_gray.shape[0]/he_image_gray_small.shape[0]
s_x = he_image_gray.shape[1]/he_image_gray_small.shape[1]


scale_matrix = np.asarray([[s_x, 0, 0],[0, s_y, 0], [0, 0, 1]])
print("scale_matrix")
print(scale_matrix)

scale_matrix_inverse = np.linalg.inv(scale_matrix)

scaled_M = np.matmul(np.matmul(scale_matrix, M), scale_matrix_inverse)

pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,scaled_M)


cv2.polylines(canvas_large,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
perspectiveM_large = cv2.getPerspectiveTransform(np.float32(dst),pts)

found_large = cv2.warpPerspective(he_image_gray,perspectiveM_large,(w,h))
cv2.imwrite("found_full_res.tif", found_large)

# he_image_recolor = cv2.cvtColor(he_image_flip, cv2.COLOR_BGR2RGB)
# he_aligned = cv2.warpPerspective(he_image_recolor,perspectiveM_large,(w,h))
he_aligned = cv2.warpPerspective(he_image_flip,perspectiveM_large,(w,h))
#cv2.imwrite("he_aligned.tif", he_aligned)
with open(r"he_aligned.pickle", "wb") as output_file:
    pickle.dump(he_aligned, output_file)

#he_aligned = cv2.normalize(he_aligned, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

writing_ome_tif(FRAMES = he_aligned,
                subresolutions = 8,
                dapi_image_level=0,
                outfile_name = "he_aligned.ome.tif",
                isRGB=False,
                dtype='uint8',
                channel_names= ['blue','green','red'],
                make_thmubnail=False)

writing_ome_tif(FRAMES = he_aligned,
                subresolutions = 8,
                dapi_image_level=0,
                outfile_name = "he_aligned.imageJ.ome.tif",
                isRGB=True,
                dtype='uint8',
                channel_names= ['blue','green','red'],
                make_thmubnail=False)

#cv2.imwrite("Dapi_unit8_full_res.tif", dapi_image_scale_0_255)
#found_large = cv2.normalize(found_large, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
#dapi_image_scale_0_255 = cv2.normalize(dapi_image_scale_0_255, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

channels = [dapi_image_scale_0_255, found_large]

# print("Reading in pickle file")
# with open('FRAMES.pickle', 'rb') as fp:
#     channels = pickle.load(fp)

# FRAMES = [dapi_image_scale_0_255, found_large]
# OUT_NAME = "Dapi_stacked_with_HE_aligned.tiff"
# print("Writing", len(FRAMES), "frames to", OUT_NAME)
# # with tifffile.TiffWriter(OUT_NAME, imagej=True) as tiff:
# with tifffile.TiffWriter(OUT_NAME) as tiff:
#     tiff.save(FRAMES[0])
#     for img in FRAMES[1:]:
#         tiff.save(img, contiguous = False)
# print("Done")

# found_large is grayscale. If we want to save the HE_aligned with the dapi images as a series of images in one file. then we need to convert the dapi to RGB from grayscale. Warning this will drastically increase that files size. It is not recommended for images that are already large.
# dapi_image_scale_0_255_rgb = cv2.cvtColor(dapi_image_scale_0_255, cv2.COLOR_GRAY2RGB) #we need to specify the number of channels below when writting multiple BigTIFF files to a single OME-TIF file so we convert dapi to 3 color which will make the file larger.
# FRAMES = [dapi_image_scale_0_255_rgb, he_aligned]
# shape = ( len(FRAMES), 1, 3,FRAMES[0].shape[0],FRAMES[0].shape[1]) #need to use different shape because color requires 2 channels instead of 1 grayscale.

with open(r"channels.pickle", "wb") as output_file:
    pickle.dump(channels, output_file)
#with open('channels.pickle', 'rb') as fp:
#    channels = pickle.load(fp)

writing_ome_tif(FRAMES = channels,
                subresolutions = 8,
                dapi_image_level=0,
                outfile_name = "Dapi_stacked_with_HE_aligned.ome.tif",
                dtype='uint8',
                channel_names = ['dapi','gray_HE'],
                make_thmubnail=False)


