# ArtificialEye
Within the masters course Active Assisted Living and in hinsight to the cybathlon 2024, we want to develop an algorithm that can run on a rasperry pi to detect within two rows of seats which chair is still available. Via a camera module and through the YOLO V8 package, occupied seats (by humans or backpacks) and empty seats will be detected. Feedback mechanism will guide the person to the object.

![image](https://github.com/leonoc97/ArtificialEye/assets/130671806/2cd0bc8c-01ec-4da2-9ee0-7bad4d39d823)

![image](https://github.com/leonoc97/ArtificialEye/assets/130671806/15a1b7b9-92f4-46a9-b410-306b258bb055)

https://github.com/leonoc97/ArtificialEye/assets/130671806/65892441-dba0-44f9-a29e-a6e8cfec15b6




Configuration Log:
Configuration: [0, 0, 0, 1, 0, 1], Frequency: 28
Configuration: [0, 1, 0, 1, 0, 1], Frequency: 81
Configuration: [0, 0, 0, 1, 1, 1], Frequency: 66
Configuration: [0, 1, 0, 1, 1, 1], Frequency: 7

1v:               0 1 0 1 0 1 
Top Configurations:
Configuration: [0, 1, 0, 1, 0, 1], Frequency: 70
Configuration: [0, 0, 0, 1, 0, 1], Frequency: 36

2v:               1 0 0 0 0 0
Top Configurations:
Configuration: [1, 0, 0, 0, 0, 0], Frequency: 7
Configuration: [0, 1, 0, 0, 0, 0], Frequency: 6



3v:               0 1 1 0 0 0 
Top Configurations:
Configuration: [0, 0, 1, 0, 0, 0], Frequency: 16
Configuration: [0, 0, 1, 0, 1, 0], Frequency: 1

4v:               0 1 0 1 0 1 
Top Configurations:
Configuration: [0, 1, 0, 1, 0, 1], Frequency: 81
Configuration: [0, 0, 0, 1, 1, 1], Frequency: 66