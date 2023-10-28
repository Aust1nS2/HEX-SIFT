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
parser.add_option("-s", "--sample_name",
                  action="store", type="string", dest="sample_name",
                  help="sample name to be incorporated into the output file name. If name is 'SampleXYZ' then the output file will be named SampleXYZ.ome.tif")

(options, args) = parser.parse_args()

#he_raw_path = "/diskmnt/Projects/HTAN_analysis_2/PDAC/xenium/image_alignment/output-XETG00122__0011117__HT434P1-S1H2Fp1Us1_1__20230919__220650/opencv_sift_flann/0011117_Scan1_crop_HT434P1-S1H2Fp1Us1_1.ome.tif"
with tifffile.TiffFile(options.he_raw_path) as tif:
    he_image = tif.series[0].levels[0].asarray()

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
       #             #physical_size_x=1 / pixelsize,
       #             #physical_size_y=1 / pixelsize,
       #             physical_size_x=pixelsize,
       #             physical_size_y=pixelsize,
       #             physical_size_x_unit='µm',
       #             physical_size_y_unit='µm'
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

writing_ome_tif(FRAMES = he_image,
                subresolutions = 8,
                dapi_image_level=0,
                outfile_name = options.sample_name+".ome.tif",
                isRGB=False,
                dtype='uint8',
                channel_names= ['blue','green','red'],
                make_thmubnail=False)




