Guidelines for the use of the dataset: 

This dataset contains inertial, positioning and step data for tracks collected at Oxford from the Oxford University Hospitals NHS Foundation Trus, 
in the United Kingdom, and at Malmö University, in Malmö, Sweden. 

The **data** folder contains a CSV file called 
    **all_data_csv.csv**: this file can be used as meta-data source of information for each single track. It contains the following information: 
       subject, testID, testName: which corresponds to subject_testID, isPatient: boolean value True or False, distanceReference: total distance walked [m], 
  	    hasMotion: boolean value True or False if during the walk IMU was collected, hasSteps: boolean value True or False if during the walk step counting was collected, 
  	    hasGNSS: boolean value True or False if during the walk GNSS signal was received, device: which device was used,	distanceByApp: distance measured by the QSS algorithm,
       totSteps: total steps were taken during the walk, path curvature: 0,1, or 2 indicating a straight, gently curved or curved path, gt_type: "final" or "continuous" according to 
       which type of reference distance was being collected, country, anonimized: boolean True or False according to whether the track geographical positions were anonymized,	
       duration [s],	fs_acc: average IMU sampling frequency for a single track,	fs_gnss: average GNSS signal sampling frequency for a single track,	fs_steps: average step counting
       sampling frequency for a single track, total_gaps_time_inertial: total time in seconds where IMU data was missing, total_gaps_time_gnss: total time in seconds 
       where GNSS signal was missing given by application in the background or bad reciving quality.
    In addition, the data folder contains an additional folder for each subject, with the correspondent folder name 
    **subject_n** with n as the subject number. Inside the "subject_n" folder there will be additional folders for each track recorded by the subject named 
        **n_x** with n being the subject number and x being the track number. Inside the n_x folder, 5 (or 6) CSV files characterize the track. 
            - **events.csv**: contains information about the **signalStart **(always set as 0ms), the **testStart** flag: 
            This flag turns to True at a certain moment in time (ms) when a maximum value of 15m is reached by the confidence interval of the GNSS sample. 
            And the testEnd flag, defining the duration in ms of the track
            - **motion.csv**: contains accelerometer and rotation rate data in the columns respectively: **accelX, accelY,	accelZ,	accelWithGX,	accelWithGY,	
            accelWithGZ,	rotRateAlpha,	rotRateBeta,	rotRateGamma.** The **interval** column represents sampling frequency in ms. While the first **ms** column 
            contains the milliseconds from signalStart.
            - **orientation.csv**: contains orientation of the device in the columns respectively: **alpha, beta, gamma** and the milliseconds from signal start **ms**. 
            - **positions.csv**: contains information from the GNSS signal. **ms** is the column for the ms passed. 
              The value of zero corresponds to the same ms of the flag testStart in the events.csv . 
              Location information are in the columns **latitude, longitude, altitude**. The column **confInterval** is the confidence interval reported by the GNSS system. 
              The column **heading** is the value of heading of the sample, while the column **speed** is the sample-wise speed computed by the system.
            - **steps.csv**: contains information of the step counter embedded in the smartphone. The columns are respectively: **ms** milliseconds from the start; 
              **steps**	incremental number of steps taken, **startDate,	endDate** as ms intervals when those steps were taken.
            - **reference_cont_distance.csv**: this file includes the continuous incremental distance information. The file has two columns: **ms** time in milliseconds 
              from signalStart, and **distance** incremental value of distance in meters. 
        
          
    
