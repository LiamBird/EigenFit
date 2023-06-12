def convert_thermofisher_map(directory, filename_output=None):
    import os
    import glob
    import numpy as np
    from tqdm import notebook
    import re
    
    allfiles = glob.glob(os.path.join(directory, "*.csv"))
    if filename_output==None:
        filename_output = "concatenated_map_data"
        
    save_name = os.path.join(directory, filename_output+".txt")
    if os.path.isfile(save_name):
        print("Concatenated map available at: "+save_name)
    
    else:
        map_data = []
        for file in notebook.tqdm(allfiles):
            x_name, y_name = os.path.split(file)[-1].strip(".CSV").split("_")
            spectrum_read = np.loadtxt(file, delimiter=",")
            map_data.append(np.hstack((np.full((spectrum_read.shape[0], 1), int(re.findall("\d+", x_name)[0])),
                   np.full((spectrum_read.shape[0], 1), int(re.findall("\d+", y_name)[0])),
                   np.loadtxt(file, delimiter=",")
                  ))
                           )

        np.savetxt(os.path.join(directory, filename_output+".txt"), np.vstack((map_data)))
        print("Concatenated map saved to "+save_name)