import os
from datetime import datetime
#os.system('ls -l')
def simple_log(l):
    print(f"{datetime.now()}: {l}")

def awsdl(source, dest):
    os.system(f'aws s3 cp --no-sign-request {source} {dest}')

def download_nsd_main():
    # Download Experiment Infos
    simple_log("downloading mat")
    awsdl("s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat", "nsddata/experiments/nsd/")
    # os.system('aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat nsddata/experiments/nsd/')
    simple_log("downloading stim info pkl")
    awsdl("s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl", "nsddata/experiments/nsd/")
    # os.system('aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl nsddata/experiments/nsd/')

    # Download Stimuli
    simple_log("Downloading hdf5")
    awsdl("s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5", "nsddata_stimuli/stimuli/nsd/")
    # os.system('aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 nsddata_stimuli/stimuli/nsd/')

    # Download Betas
    simple_log("Starting betas")
    for sub in [1,2,5,7]:
        simple_log(f"beta {sub}")
        for sess in range(1,38):
            simple_log(f"Session {sess}")
            os.system('aws s3 cp --no-sign-request s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))

    # Download ROIs
    for sub in [1,2,5,7]:   
        simple_log(f"Download ROI {sub}")
        awscmd = 'aws s3 cp --recursive --no-sign-request s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub)
        os.system(awscmd)

if __name__ == "__main__":
    last_cwd = os.getcwd()
    nsd_data_root = "/home/matt/diffusing/brain-diffuser/data"
    os.chdir(nsd_data_root)
    download_nsd_main()
    os.chdir(last_cwd)