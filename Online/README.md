## Notes
1. Inherit the "TriggerUnit" and "AmpDataClient" classes in AmpInterface.py to build your own amplifier interface.   
Then you should add the dataClient type to OnlineSystem.py line 88 and the triggerUnit to Stimulator.py line 348.   
We already implemented interface for Neuracle wireless EEG amplifier.
2. The class "Stimulator" in Stimulator.py controls the display of visual stimuli while "TrainingController" and "TestingController" 
in Controller.py control the training phase and testing phase respectively. 