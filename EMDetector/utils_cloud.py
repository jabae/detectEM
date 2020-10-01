"""
Utils for cloudvolume IO
"""

# Import necessary packages
from cloudvolume import CloudVolume


# Upload volume to cloud
def upload_cloud(cloud_dir, volume, layer_type, dtype, resolution, volume_size, voxel_offset=[0,0,0], prov_description=""):
    """
    cloud_dir : Cloud directory to upload
    volume : Volume to upload
    layer_type : 'image' or 'segmentation'
    dtype : Has to match volume's data type
    resolution : Resolution of voxel
    volume_size : Volume size
    voxel_offset : Volume offset
    prov_description : Provenance description 
    """

    info = CloudVolume.create_new_info(
        num_channels = 1,
        layer_type = layer_type, # 'image' or 'segmentation'
        data_type = dtype, # can pick any popular uint
        encoding = 'raw', # other option: 'jpeg' but it's lossy
        resolution = resolution, # X,Y,Z values in nanometers
        voxel_offset = voxel_offset, # values X,Y,Z values in voxels
        chunk_size = [ 128, 128, 1 ], # rechunk of image X,Y,Z in voxels
        volume_size = volume_size, # X,Y,Z size in voxels
    )


    vol = CloudVolume(cloud_dir, parallel=True, progress=True, cdn_cache=False, info=info)
    vol.provenance.description = prov_description
    vol.provenance.owners = ['jabae@princeton.edu'] # list of contact email addresses

    vol.commit_info() # generates gs://bucket/dataset/layer/info json file
    vol.commit_provenance() # generates gs://bucket/dataset/layer/provenance json file

    vol[:,:,:] = volume


# Extract cloudvolume chunk
def read_cloud_chunk(cloud_dir, chunk_begin, chunk_end, mip=0, parallel=True, progress=True):
    """
    cloud_dir : Cloud directory
    chunk_begin : Start index
    chunk_end : End index
    parallel : True for parallel download
    progress : Show progress
    """

    vol = CloudVolume(cloud_dir, mip=mip, parallel=parallel, progress=progress)
    chunk_range = [slice(chunk_begin[i], chunk_end[i]) for i in range(3)]

    return vol[chunk_range][:,:,:,0]


# Write cloudvolume chunk
def write_cloud_chunk(cloud_dir, volume, chunk_begin, chunk_end, mip=0, parallel=True, progress=True):
    """
    cloud_dir : Cloud directory
    volume : Chunk to upload
    chunk begin : start index
    chunk end : start index
    parallel : True for parallel download
    progress : Show progress
    """

    vol = CloudVolume(cloud_dir, mip=0, parallel=parallel, progress=progress)
    chunk_range = [slice(chunk_begin[i], chunk_end[i]) for i in range(3)]

    # Write chunk
    vol[chunk_range] = volume.astype(vol.dtype)
