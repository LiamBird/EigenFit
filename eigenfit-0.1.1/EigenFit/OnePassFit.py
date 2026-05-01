class OnePassFit(object):
    def __init__(self, SeriesDimensionReduction_obj, save_path=None, labels_to_fit="all", 
                 peaks_to_fit=None, center_tolerance=10, sigma_value=2, sigma_max=20,
                 _diagnostic_run=False, _diagnostic_x=10, _diagnostic_y=12):
        
########## Set-up starts ##########      

        if _diagnostic_run == True:
            print("Diagnostic run only")
            self._diagnostic_run = True
        else:
            self._diagnostic_run = False
            
        from lmfit.models import LinearModel, LorentzianModel
        from lmfit import Parameters
        from make_lorentzian_params import make_lorentzian_params
        from tqdm import notebook
        from datetime import datetime
        
        self.x_extent = SeriesDimensionReduction_obj.common.x_extent
        self.y_extent = SeriesDimensionReduction_obj.common.y_extent
        self.shift_extent = SeriesDimensionReduction_obj.common.shift_extent
        
        self.map_normalised = SeriesDimensionReduction_obj.map_normalised
        self.shift = SeriesDimensionReduction_obj.common.shift
        
        self._series_label = SeriesDimensionReduction_obj._series_label
        setattr(self, self._series_label, vars(SeriesDimensionReduction_obj)[self._series_label])
        self.fitted_values = {}
        self.fitted_spectra = {}
        self.fit_times = {}    
        
        if peaks_to_fit == None:
            self.peaks_to_fit = [152, 202, 218, 248, 398, 438, 450, 473, 504, 745]
        else:
            self.peaks_to_fit = peaks_to_fit
        
        if save_path != None:
            self._save_path = save_path
        elif save_path == None:
            ## Copying path from DimRed:      
            dimred_path = SeriesDimensionReduction_obj._save_path
            if "Preprocessed" in dimred_path:
                save_path = "\\".join([seg for seg in dimred_path.split("\\")[:dimred_path.split("\\").index("Preprocessed")]]) 
            else:
                save_path = os.path.split(dimred_path)[0]
            self._save_path = os.path.join(save_path, "one_pass_fit")
        try:
            os.makedirs(self._save_path)
        except:
            if os.path.isdir(self._save_path):
                if self._diagnostic_run == True:
                    print("Path already exists")
            else:
                print("Failed to make save path in requested directory: making save path in current directory")
                os.makedirs("one_pass_fit")
                self._save_path = "one_pass_fit"
            
        if labels_to_fit == "all":
            labels_to_fit = vars(self)[self._series_label]
        else:
            labels_to_fit = list(labels_to_fit)
        self.labels_to_fit = labels_to_fit    
########## Set-up ends ########## 


        for label in labels_to_fit:
            print(label)
## Scenario 1: The data has been fitted, and the values are available and saved in the path
            if str(label) in os.listdir(self._save_path) :
                print("label found in path")
                if "fitted_lorentzian_params.npy" in os.listdir(os.path.join(self._save_path, str(label))):
                    if self._diagnostic_run == True:
                        print("Reloading previous files")
                    self.fitted_values.update([(label,np.load(os.path.join(self._save_path, str(label), "fitted_lorentzian_params.npy"), allow_pickle=True).item())])
                    self.fitted_spectra.update([(label, np.load(os.path.join(self._save_path, str(label), "fitted_spectra.npy"), allow_pickle=True))])
                    self.fit_times.update([(label, np.load(os.path.join(self._save_path, str(label), "time_to_fit.npy"), allow_pickle=True))])
                    
## Scenario 2: The data has been partly fitted, and the reloaded values are incomplete
                    if any(np.isnan(self.fit_times[label]).flatten()):
        
                        model = LinearModel()
                        for peak in self.peaks_to_fit:
                            model += LorentzianModel(prefix="peak_{}_".format(peak))
                        self.params = make_lorentzian_params(model, center_tolerance=center_tolerance, sigma_value=sigma_value, sigma_max=sigma_max)
        
        
                        if self._diagnostic_run == True:
                            print("Resuming previous incomplete fit")
                            spectrum = SeriesDimensionReduction_obj.map_normalised[label][_diagnostic_x+1, _diagnostic_y, :]
                            shift = SeriesDimensionReduction_obj.common.shift
                            self._1PF_fitting(label=label, xi=_diagnostic_x+1, yi=_diagnostic_y, spectrum=spectrum, shift=shift)
                            self.xnan, self.ynan = np.nonzero(np.isnan(self.fit_times[label]))
                            
                        else:
                            xnan, ynan = np.nonzero(np.isnan(self.fit_times[label]))
                            skipped_pos = np.vstack((xnan, ynan)).T

                            for idx in notebook.tqdm(np.arange(skipped_pos.shape[0])):
                                xi, yi = skipped_pos[idx, :]
                                spectrum = SeriesDimensionReduction_obj.map_normalised[label][xi, yi, :]
                                shift = SeriesDimensionReduction_obj.common.shift
                                self._1PF_fitting(label=label, xi=xi, yi=yi, spectrum=spectrum, shift=shift)
                        
## Scenario 3: The data has not been fitted
            if str(label) not in os.listdir(self._save_path) or all([str(label) in os.listdir(self._save_path),
                                                                       "fitted_lorentzian_params.npy"
                                                                       not in os.listdir(os.path.join(self._save_path, str(label)))]):
                
                from make_lorentzian_params import make_lorentzian_params
                
                try:
                    os.makedirs(os.path.join(self._save_path, str(label)))
                except:
                    pass
                
                model = LinearModel()
                for peak in self.peaks_to_fit:
                    model += LorentzianModel(prefix="peak_{}_".format(peak))
                self.params = make_lorentzian_params(model, center_tolerance=center_tolerance, sigma_value=sigma_value, sigma_max=sigma_max)
                
                self.fitted_values.update([(label, 
                                            dict([(peak, 
                                                   dict([(name, np.full((self.x_extent, self.y_extent), np.nan))
                                                        for name in ["amplitude", "center", "intensity", "sigma"]]))
                                                  for peak in self.peaks_to_fit])
                                           )])
                self.fitted_values[label].update([(name, np.full((self.x_extent, self.y_extent), np.nan)) 
                                                         for name in ["slope", "intercept"]])
                
                self.fitted_spectra.update([(label, np.full((self.x_extent, self.y_extent, self.shift_extent), np.nan))])
                self.fit_times.update([(label, np.full((self.x_extent, self.y_extent), np.nan))])
                
                
                if self._diagnostic_run == True:
                    xi = _diagnostic_x
                    yi = _diagnostic_yi
                    spectrum = SeriesDimensionReduction_obj.map_normalised[label][xi, yi, :]
                    shift = SeriesDimensionReduction_obj.common.shift
                    self._1PF_fitting(model, label, xi, yi, spectrum, shift) 
                else:    
                    xi = np.arange(self.x_extent)
                    yi = np.arange(self.y_extent)
                    xycoords = np.vstack((np.meshgrid(xi, yi)[0].flatten(),
                                          np.meshgrid(xi, yi)[1].flatten())).T
                    for idx in notebook.tqdm(np.arange(xycoords.shape[0])):
                        xi, yi = xycoords[idx, :]
                        spectrum = SeriesDimensionReduction_obj.map_normalised[label][xi, yi, :]
                        shift = SeriesDimensionReduction_obj.common.shift
                        self._1PF_fitting(model, label, xi, yi, spectrum, shift)                        
                        
####################        
    def _1PF_fitting(self, model, label, xi, yi, spectrum, shift):
        from lmfit.models import LinearModel, LorentzianModel
        from lmfit import Parameters
        from make_lorentzian_params import make_lorentzian_params
        from tqdm import notebook
        from datetime import datetime
        
        start_time = datetime.now()
        model_fit = model.fit(data=spectrum,
                              x=shift,
                              params=self.params)
        end_time = datetime.now()

        self.fitted_spectra[label][xi, yi] = model_fit.best_fit
        self.fitted_values[label]["slope"][xi, yi] = model_fit.best_values["slope"]
        self.fitted_values[label]["intercept"][xi, yi] = model_fit.best_values["intercept"]
        for peak in self.peaks_to_fit:
            self.fitted_values[label][peak]["amplitude"][xi, yi] = model_fit.best_values["peak_{}_amplitude".format(peak)]
            self.fitted_values[label][peak]["center"][xi, yi] = model_fit.best_values["peak_{}_center".format(peak)]
            self.fitted_values[label][peak]["sigma"][xi, yi] = model_fit.best_values["peak_{}_sigma".format(peak)]
            self.fitted_values[label][peak]["intensity"][xi, yi] = model_fit.best_values["peak_{}_amplitude".format(peak)]/model_fit.best_values["peak_{}_sigma".format(peak)]/np.pi                    
        self.fit_times[label][xi, yi] = 1e6*(end_time-start_time).seconds+(end_time-start_time).microseconds

        if self._diagnostic_run == True:
            print("Fitted x={}, y={}".format(xi, yi))
            print(os.path.join(self._save_path, str(label), "fitted_lorentzian_params.npy"))
        np.save(os.path.join(self._save_path, str(label), "fitted_lorentzian_params.npy"),
                self.fitted_values[label], allow_pickle=True)
        np.save(os.path.join(self._save_path, str(label), "fitted_spectra.npy"),
                self.fitted_spectra[label], allow_pickle=True)
        np.save(os.path.join(self._save_path, str(label), "time_to_fit.npy"),
                self.fit_times[label], allow_pickle=True)
####################

    def view_fit_results(self):
        from ipywidgets import interact, IntSlider
        f, ax =  plt.subplots()

        def update(xi, yi, l_idx):
            ax.cla()
            label = self.labels_to_fit[l_idx]
            ax.plot(self.shift, self.map_normalised[label][xi, yi], label="Raw")
            ax.plot(self.shift, self.fitted_spectra[label][xi, yi], label="Fitted")
            ax.set_title(label)
            ax.legend()

        interact(update, l_idx=IntSlider(min=0, max=len(self.labels_to_fit)-1, step=1),
                         xi=IntSlider(min=0, max=self.x_extent-1),
                         yi=IntSlider(min=0, max=self.y_extent-1))
        
        return f, ax
