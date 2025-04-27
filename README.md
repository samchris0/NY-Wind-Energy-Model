# NY-Wind-Energy-Model
The purpose of this model is to...


# Update from Drew (Sunday April 27)
Hey Sam, quick update on everything new to the repo 

### **`IMPORTANT`** What you'll need to do first to use the repo

Once you pull, all you should have to do is download the zipped data file from [here](https://drive.google.com/drive/folders/1kCGSDBHP06SO5-z203F4gQVOZKqsNW68?usp=sharing) and replace the existing data folder with it. It'll be at least a few gb of storage since it contains the full unfiltered history but there shouldn't be any issues with accidentally committing the data to github (fingers crossed)

I figure this google drive folder can also be used for our presentation slides and final report once we get there 

- ## **New directory structure**
  -  hopefully pretty straightforward to follow.
  -  I dumped a lot of the original scripts you wrote in the preprocessing folder for now and we can sort out what is important to use again once the prediction model is working better 
-  ## **New Files**
   -  `PrepareSensorData.py`
      -  This has the retrieve_data, fill_missing_hours, get_mag_df, get_coordinate_dicts functions which are independent from the actual submodular optimization process.
      -  The functionality is the same although I had to change how things are accessed when I change how the sensor data is stored.
      -  I did however modify df_mag to forward fill all the NaN values to see if that fixed the LTSM
   - `preprocessing/filterHistoricalForecast_v2.py`
      -  I modified your filter function to work with the new data format.
      -  It'll first check if the filtered data exists in your `data/filtered_historicalForecasts` folder and otherwise filters it from the `data/unfiltered_historicalForecast2024.csv` file.
   -  `OptimizedSelection.py`
      -  This has make_kernel, mutual_info_gain, greedy_select, and a new dispatcher function (get_optimized_sensors) that runs the submodular optimization process
      -  No big changes here either. get_optimized_sensors returns a list of k sensor_ids which are unique and mapped to each sensor in all of NY (using lat_dict and lon_dict)
   -  `LazySelections.py`
      -  New functions I wrote to generate the three control sensor selections we're going to compare to the submodular optimization result
   -  `sensorSelection.py`
      -  Contains dispatcher function `get_sensor_selections` that calculates the optimized sensor selection as well as the three lazy sensor selections
      -  Since the optimized sensor selection and lazy geographic selections can take a while it caches results for given k and dist in `data/selection_cache.csv`
      -  returns a dictionary of sensor ids for each selection heuristic along with `df_mag_sqrt` and coordinate dictionaries 
   -  `preprocessing/general_preprocessing_tools.ipynb`
      -  This notebook has some functions I used to cleanup the historical sensor data csvs.
      -  I don't think we'll really need anything in here again now that the data is configured
   
-  ## **Unneeded Files** 

    Kept your code just in case but have new versions of these using the new data format   
   -  `MIOpt_legacy.py`
      -  We should use the pipeline setup in `test_pipe.ipynb` going forward
   -  `preprocessing/filterHistoricalForecease.py`
      -  We an use `preprocessing/filterHistoricalForecase_v2.py` going forward
-  ## **Status of `test_pipe.ipynb`**
   -  Starts by calling `get_sensor_selections` to compute/retrieve the selected sensor ids 
   -  I was able to get the LTSM model to compile but the prediction output is basically a straight line irregardless of the parameters I mess with
      -  **(Full disclosure I did a lot of chatGPT so can't really vouch for the result other than that it compiles)**
      -  Made some changes to `makeModel.py` to get it to work and also had to mess with some module version and dependency issues so if it doesn't compile lmk and I can try and show you what I did to fix. 
      -  I was totally just plugging in random numbers so apologies if I messed things up haha (but at least it runs now)



