import numpy as np
import cv2
import tifffile
import skimage
from optparse import OptionParser
from ome_types import from_tiff, from_xml, to_xml, model
from ome_types.model.simple_types import UnitsLength
import sys
import scipy
import scipy.ndimage

parser = OptionParser()
parser.add_option("-i", "--image",
                  action="store", type="string", dest="image",
                  help="path to the raw input H&E image to be aligned to dapi image")
parser.add_option("-s", "--size",
                  action="store", type="string", dest="size", default="",
                  help="This option can be used to directly specify the size of the output image when a blank image is to be written as a .ome.tif file. This accepts comma separate values as input in an x,y format. x= width of the image in pixels when it is opened in qupath. y = height of the image in pixels when it is opened in qupath. (E.g. 100,500 results in the generation of an image of width=100 pixels, height=500 pixels). Warning: if you use this option then the metadata of the all black image will not be accurate regarding the pixel to micron distance ratio used to make and show scale bars.")
parser.add_option("-b", "--blank_write",
                  action="store_true", dest="blank_write", default=False,
                  help="When specified this option results in the generation of a blank (all black) image that annotations can be saved on top of in QuPath. Image is saved as an ome.tif file. Comma separated x,y values for the size of the black image to be generated. x= width of the image when it is opened in qupath. y = height of the image when it is opened in qupath. The image size is determined by either the use of the -s (--size) flag or the -i (--image) flag. See those options for a description of their use.")
parser.add_option("-w", "--workers",
                  action="store", type="int", dest="workers", default=8,
                  help="The number of threads that will be used to save the image. Default is 8.")              
parser.add_option("-a", "--annotation_mask",
                  action="store", type="string", dest="annotation_mask", default="",
                  help="black image with the annotation drawn_on_it. Annotation color should be bright red, green, blue, or yellow. The annotation is converted from a RGB image to grey scale and then an intensity of 50 is used as the cutoff.")
parser.add_option("-n", "--output_name",
                  action="store", type="string", dest="output_name", default="output_image",
                  help="name of the output file. This script will save the image as an .ome.tif image.")
parser.add_option("-d", "--output_directory",
                  action="store", type="string", dest="output_dir", default="./",
                  help="path to output folder that the image will be saved to. ")
# parser.add_option("-p", "--pixel_size",
#                   action="store", type="string", dest="pixel_size", default="./",
#                   help="The size of a pixel in microns. This option is only used if specifying the output blank image size using the --size option. Otherwise the script pulls this from the example input image.")

#python3 $austin/tools/HEX-SIFT/write_annotation_masked_image_v1.py -b -i ID_0033854_Scan1.qptiff -n HT179C1-T1Fp3L5U1_blank_image_for_annotation -d ./
#python3 $austin/tools/HEX-SIFT/write_annotation_masked_image_v2.py -i ID_0033854_Scan1.qptiff -a HT179C1-T1Fp3L5U1_annotated_blank.ome.tif -n HT179C1-T1Fp3L5U1_crop -d ./ -w 10

def writing_ome_tif(FRAMES = np.zeros((3,512,512)),
                    subresolutions = 8,
                    dapi_image_level=0,
                    outfile_name = "test.ome.tif",
                    isRGB=False,
                    dtype='uint8',
                    channel_names=[],
                    make_thmubnail=False,
                    num_threads=8,
                    pixel_resolution = 10000,
                    pixel_size_unit = 'µm'):
                    
    # dapi_image_dict = {0:0.2125,
    #                    1:0.4250,
    #                    2:0.8500,
    #                    3:1.7000,
    #                    4:3.4000,
    #                    5:6.8000,
    #                    6:13.6000,
    #                    7:27.2000}
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
    #pixelsize = dapi_image_dict[dapi_image_level]  # micrometer #this is what is used in HEX-SIFT
    pixelsize = pixel_resolution
    print(pixelsize)
    print(pixel_size_unit)
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
            'PhysicalSizeXUnit': pixel_size_unit,
            'PhysicalSizeY': pixelsize,
            'PhysicalSizeYUnit': pixel_size_unit,
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
            maxworkers=num_threads #2 #This is the number of threads to use when saving
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

def write_blank_image(size = (100, 100), num_workers=8, output_resolution = 10000, pixel_size_unit = 'µm', out_name = "blank_image_for_annotation"): #The default pixel_resolution is set so that it will always be wrong because it is a complete guess and I don't want anyone using it.
    height = size[0]
    width = size[1]
    zero_array = np.zeros((3, height, width), dtype=float)
    print(output_resolution)
    print(pixel_size_unit)
    writing_ome_tif(FRAMES = zero_array,
                    subresolutions = 8,
                    dapi_image_level=0,
                    outfile_name = out_name+".ome.tif",
                    isRGB=True, #
                    dtype='uint8',
                    channel_names= ['blue','green','red'],
                    make_thmubnail=False, #do not make the thumbnail. I don't know why but for some reason this causes inconsistencies in which platforms can load the image when trying to load it into qupath/XeniumExplorer/imageJ, so for now I never do this.
                    num_threads=num_workers,
                    pixel_resolution = output_resolution,
                    pixel_size_unit = pixel_size_unit)

(options, args) = parser.parse_args()
threads = options.workers
size = options.size
input_image_path = options.image
output_file = options.output_dir+"/"+options.output_name
print(output_file)
if options.blank_write:
    if options.size != "":
        print("writing blank (all black) image based on specified size. The output image file metadata will not be accurate as no reference pixel to micron conversion is available.")
        width = int(options.size.strip().split(",")[0])
        height = int(options.size.strip().split(",")[1])
        size = (height,width)
        write_blank_image(size = size, num_workers=threads, out_name = output_file)
        sys.exit("blank image has been written for annotation. Annotate in QuPath and then re-run the script with the necessary masking image and parameters.")
    elif input_image_path != "":
        print("writing blank (all black) image based on the size of the provided image.")
        if ".ome.tif" in input_image_path:
            print("inut file is a .ome.tif file")
            ome_image = tifffile.TiffFile(input_image_path)
            ome_meta_list = ome_image.ome_metadata.split(" ")
            for value in ome_meta_list:
                # here we are assuming that PhysicalSizeY is equivalent to PhysicalsizeX which if we are working with the raw output that should always be the case.
                if "PhysicalSizeY" in value:
                    # here we are assuming that 
                    physicalSize_split = value.strip().split('"')
                    pysicalSizeY = float(physicalsize_split[1])
                elif "PhysicalSizeYUnit" in value:
                    physicalSize_split = value.strip().split('"')
                    pysicalSizeYUnit = str(physicalSize_split[1])
                elif "PhysicalSizeX" in value:
                    physicalSize_split = value.strip().split('"')
                    pysicalSizeX = str(physicalSize_split[1])
                elif "PhysicalSizeXUnit" in value:
                    physicalSize_split = value.strip().split('"')
                    pysicalSizeXUnit = str(physicalSize_split[1])
                elif "SizeX" == value:
                    pixelSize_split = value.strip().split('"')
                    X_length_pixels = str(pixelSize_split[1])
                elif "SizeY" == value:
                    pixelSize_split = value.strip().split('"')
                    Y_length_pixels = str(pixelSize_split[1])
            if pysicalSizeX != pysicalSizeY:
                print("the ome tif has different X and Y voxel resolutions (pixels per micron is different between them). Proceeding with the Y voxel resolution. Use of the output image may not work in downstream analysis.")
            output_pixesize = pysicalSizeY
            output_pixelsize_unit = pysicalSizeYUnit
            print(output_pixesize)
            print(output_pixelsize_unit)
            input_image = np.asarray(tifffile.imread(input_image_path, is_ome=False, level=0))
            input_size = input_image.shape
        else:
            print("input file is not an ome.tif file")
            input_image = np.asarray(tifffile.imread(input_image_path, is_ome=False, level=0)) #assume the image is not pyramidal since the 
            input_size = input_image.shape #get image shape in pixels from the image itself since image sizes are stored differently in different formats and this part needs to be correct no matter what. 
            # tested on the following input file formats is a .qptiff (from Phenocycler) or a .ome.tif (from QuPath - crop of image from Phenocycler)
            # input_image_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/HT061P1-S1P1A1L1U1/testing_blank_crop/ID_0034249_Scan1.qptiff"
            # qupath_ome_image_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/HT061P1-S1P1A1L1U1/HT061P1-S1P1A1L1U1.ome.tif"
            resolution_parts = tifffile.TiffFile(input_image_path).pages.pages[0].tags['XResolution'].value
            # only when resolutionunit='CENTIMETER',
            pixel_resolution = resolution_parts[1]/resolution_parts[0]
            output_pixesize = pixel_resolution*1e4
            output_pixelsize_unit = 'µm' #this is an assumption but it should be true for all 20X and 40X images generated in out lab so I am hard-coding it.
            print(output_pixesize)
            print(output_pixelsize_unit)
        # dapi_image_path = "/diskmnt/primary/Xenium/data/data_reprocess_xeniumranger_v2.0/20241108__214933__20241108_V1_KRAS_HTAN_PDAC_CRC/output-XETG00122__0034249__HT061P1-S1P1A1L1U1__20241108__215027_xrv2_0_ne_5um_df_100_rn_true/outs/morphology_focus/morphology_focus_0000.ome.tif
        write_blank_image(size = input_size, num_workers=threads, output_resolution = output_pixesize, pixel_size_unit = output_pixelsize_unit, out_name = output_file)
        sys.exit("blank image has been written for annotation. Annotate in QuPath and then re-run the script with the necessary masking image and parameters.")
    sys.exit("no input template image or image size parameters provided. Exiting without generating output.")


######
#read in the input image - #this has been tested on qptiff images. It has not been tested on ome.tif images

#input_image_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/HT061P1-S1P1A1L1U1/testing_blank_crop/ID_0033854_Scan1.qptiff"
input_image = np.asarray(tifffile.imread(input_image_path, is_ome=False, level=0))
print(input_image.shape)
#(43200, 24960, 3)

#get the pixel resolution needed to properly write the image metadata
if ".ome.tif" in input_image_path:
    print("inut file is a .ome.tif file")
    ome_image = tifffile.TiffFile(input_image_path)
    ome_meta_list = ome_image.ome_metadata.split(" ")
    for value in ome_meta_list:
        # here we are assuming that PhysicalSizeY is equivalent to PhysicalsizeX which if we are working with the raw output that should always be the case.
        if "PhysicalSizeY" in value:
            # here we are assuming that 
            physicalSize_split = value.strip().split('"')
            pysicalSizeY = float(physicalsize_split[1])
        elif "PhysicalSizeYUnit" in value:
            physicalSize_split = value.strip().split('"')
            pysicalSizeYUnit = str(physicalSize_split[1])
        elif "PhysicalSizeX" in value:
            physicalSize_split = value.strip().split('"')
            pysicalSizeX = str(physicalSize_split[1])
        elif "PhysicalSizeXUnit" in value:
            physicalSize_split = value.strip().split('"')
            pysicalSizeXUnit = str(physicalSize_split[1])
        elif "SizeX" == value:
            pixelSize_split = value.strip().split('"')
            X_length_pixels = str(pixelSize_split[1])
        elif "SizeY" == value:
            pixelSize_split = value.strip().split('"')
            Y_length_pixels = str(pixelSize_split[1])
    if pysicalSizeX != pysicalSizeY:
        print("the ome tif has different X and Y voxel resolutions (pixels per micron is different between them). Proceeding with the Y voxel resolution. Use of the output image may not work in downstream analysis.")
    output_pixesize = pysicalSizeY
    output_pixelsize_unit = pysicalSizeYUnit
    print(output_pixesize)
    print(output_pixelsize_unit)
else:
    print("input file is not an ome.tif file")
    input_image = np.asarray(tifffile.imread(input_image_path, is_ome=False, level=0)) #assume the image is not pyramidal since the 
    input_size = input_image.shape #get image shape in pixels from the image itself since image sizes are stored differently in different formats and this part needs to be correct no matter what. 
    # tested on the following input file formats is a .qptiff (from Phenocycler) or a .ome.tif (from QuPath - crop of image from Phenocycler)
    # input_image_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/HT061P1-S1P1A1L1U1/testing_blank_crop/ID_0034249_Scan1.qptiff"
    # qupath_ome_image_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/HT061P1-S1P1A1L1U1/HT061P1-S1P1A1L1U1.ome.tif"
    resolution_parts = tifffile.TiffFile(input_image_path).pages.pages[0].tags['XResolution'].value
    # only when resolutionunit='CENTIMETER',
    pixel_resolution = resolution_parts[1]/resolution_parts[0]
    output_pixesize = pixel_resolution*1e4
    output_pixelsize_unit = 'µm' #this is an assumption but it should be true for all 20X and 40X images generated in out lab so I am hard-coding it.
    print(output_pixesize)
    print(output_pixelsize_unit)

#read in the image mask (an otherwise blank image with the area of interest annotated in red, green, blue, yellow)
mask_path = options.annotation_mask
#mask_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/HT061P1-S1P1A1L1U1/testing_blank_crop/HT179C1-T1Fp3L5U1_annotated_blank.ome.tif"
image_mask = np.asarray(tifffile.imread(mask_path, is_ome=False, level=0)) # assume that the image mask is an RBG image
print(image_mask.shape)
# greyscale the mask
image_mask_gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
# find the holes (threshold)
tf_mask_outline = image_mask_gray > 50
# close the holes
tf_mask = scipy.ndimage.binary_fill_holes(tf_mask_outline)
# binarize (0, 1) - think of this as a true false mask for which pixels will be retained.
tf_mask_image = tf_mask*255
tf_mask_image = tf_mask_image.astype(np.uint8)
# cv2.imwrite('tf_image_mask.tif', tf_mask_image)
tf_mask_image_invert = 255-tf_mask_image
tf_mask_image_invert = tf_mask_image_invert.astype(np.uint8)
# cv2.imwrite('tf_image_mask_invert.tif', tf_mask_image_invert)

# to each channel of the input image apply the mask
blue_masked = input_image[:,:,0]*tf_mask
green_masked = input_image[:,:,1]*tf_mask
red_masked = input_image[:,:,2]*tf_mask

# cv2.imwrite('blue_masked.tif', blue_masked)
# cv2.imwrite('green_masked.tif', green_masked)
# cv2.imwrite('red_masked.tif', red_masked)

output_masked_image = np.zeros((3,image_mask.shape[0],image_mask.shape[1])).astype(np.uint8)
output_masked_image[0,:,:] = blue_masked
output_masked_image[1,:,:] = green_masked
output_masked_image[2,:,:] = red_masked

blue_masked_white = blue_masked + tf_mask_image_invert
green_masked_white = green_masked + tf_mask_image_invert
red_masked_white = red_masked + tf_mask_image_invert

# cv2.imwrite('blue_masked_white.tif', blue_masked_white)
# cv2.imwrite('green_masked_white.tif', green_masked_white)
# cv2.imwrite('red_masked_white.tif', red_masked_white)

output_masked_image_white = np.zeros((3,image_mask.shape[0],image_mask.shape[1])).astype(np.uint8)
output_masked_image_white[0,:,:] = blue_masked_white
output_masked_image_white[1,:,:] = green_masked_white
output_masked_image_white[2,:,:] = red_masked_white

writing_ome_tif(FRAMES = output_masked_image,
                    subresolutions = 8,
                    dapi_image_level=0,
                    outfile_name = output_file+".ome.tif",
                    isRGB=True, #
                    dtype='uint8',
                    channel_names= ['blue','green','red'],
                    make_thmubnail=False, #do not make the thumbnail. I don't know why but for some reason this causes inconsistencies in which platforms can load the image when trying to load it into qupath/XeniumExplorer/imageJ, so for now I never do this.
                    num_threads=threads,
                    pixel_resolution = output_pixesize,
                    pixel_size_unit = output_pixelsize_unit)

writing_ome_tif(FRAMES = output_masked_image_white,
                    subresolutions = 8,
                    dapi_image_level=0,
                    outfile_name = output_file+"_white_background.ome.tif",
                    isRGB=True, #
                    dtype='uint8',
                    channel_names= ['blue','green','red'],
                    make_thmubnail=False, #do not make the thumbnail. I don't know why but for some reason this causes inconsistencies in which platforms can load the image when trying to load it into qupath/XeniumExplorer/imageJ, so for now I never do this.
                    num_threads=threads,
                    pixel_resolution = output_pixesize,
                    pixel_size_unit = output_pixelsize_unit)

