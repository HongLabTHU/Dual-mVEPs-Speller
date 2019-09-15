"""
To read .nex or .nex5 files, use the following code:
    import nexfile
    reader = nexfile.Reader()
    fileData = reader.ReadNexFile('C:\\Data\\file.nex')
    fileData1 = reader.ReadNexFile('C:\\Data\\file.nex5')

If your files are larger than a few MB, use numpy version of the reader:
    import nexfile
    reader = nexfile.Reader(useNumpy=True)
    fileData = reader.ReadNexFile('C:\\Data\\LargeFile.nex')

To write .nex file, use this code:
    timestampFrequency = 50000
    writer = nexfile.NexWriter(timestampFrequency)
then, add variable data using Add... methods in NexWriter class
(see method doc strings below for more info):
    writer.AddContVarWithSingleFragment('cont1', 0, 10000, [5, 6, 7, 8])
    writer.AddContVarWithSingleFragment('cont2', 0, 10000, [9, 10, 11, 12])
then, use WriteNexFile method:
    writer.WriteNexFile('C:\\Data\\python.nex')

If your files are larger than a few MB, use numpy version of the NexWriter:
    import nexfile
    import numpy as np
    timestampFrequency = 50000
    writer = nexfile.NexWriter(timestampFrequency, useNumpy=True)
    writer.AddNeuron('neuron1', np.array([1, 2, 3, 4]))
    writer.AddContVarWithSingleFragment('cont1', 2, 10000, np.array([5, 6, 7, 8]))
    writer.WriteNexFile('C:\\Data\\pythonWithFloatContValues.nex5', 1)
"""

import sys
import os
import struct
import array
import json
import numbers


class NexFileVarType:
    """
    Constants for .nex and .nex5 variable types
    """
    NEURON = 0
    EVENT = 1
    INTERVAL = 2
    WAVEFORM = 3
    POPULATION_VECTOR = 4
    CONTINUOUS = 5
    MARKER = 6


class Reader(object):
    """
    Nex file reader class
    """
    def __init__(self, useNumpy=False):
        """
        Constructor
        :param useNumpy: option to use numpy to read data arrays.
        """
        self.theFile = None
        self.fileData = None
        self.useNumpy = useNumpy
        self.fromTicksToSeconds = 1
        
    def ReadNex5File(self, filePath):
        """ 
        Reads data from .nex5 file.
        :param filePath: full path of file
        :return: file data
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == '.nex':
            return self.ReadNexFile(filePath)
        self.fileData = {}
        self.theFile = open(filePath, 'rb')

        # read file header
        self.fileData['FileHeader'] = self._ReadNex5FileHeader()
        self.fileData['Variables'] = []

        # read variable headers and create variables
        for varNum in range(self.fileData['FileHeader']['NumVars']):
            var = {'Header': self._ReadNex5VarHeader()}
            self.fileData['Variables'].append(var)

        # read variable data
        self._ReadData()

        # read metadata
        metaOffset = self.fileData['FileHeader']['MetaOffset']
        if metaOffset > 0:
            self.theFile.seek(0, os.SEEK_END)
            size = self.theFile.tell()
            if metaOffset < size:
                self.theFile.seek(metaOffset)
                metaString = self.theFile.read(size - metaOffset).decode('utf-8').strip('\x00')
                metaString = metaString.strip()
                try:
                    self.fileData['MetaData'] = json.loads(metaString)
                except Exception as error:
                    print('Invalid file metadata: ' + repr(error))

        self.theFile.close()
        return self.fileData

    def ReadNexFile(self, filePath):
        """
        Reads data from .nex file.
        :param filePath:
        :return: file data
        """
        extension = os.path.splitext(filePath)[1].lower()
        if extension == '.nex5':
            return self.ReadNex5File(filePath)

        self.fileData = {}
        self.theFile = open(filePath, 'rb')

        self.fileData['FileHeader'] = self._ReadFileHeader()
        self.fileData['Variables'] = []

        for varNum in range(self.fileData['FileHeader']['NumVars']):
            var = {'Header': self._ReadVarHeader()}
            self.fileData['Variables'].append(var)

        self._ReadData()

        self.theFile.close()
        return self.fileData

    def _ReadData(self):
        for var in self.fileData['Variables']:
            self.theFile.seek(var['Header']['DataOffset'])
            varType = var['Header']['Type']
            if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
                self._ReadTimestamps(var)
            elif varType == NexFileVarType.INTERVAL:
                self._ReadIntervals(var)
            elif varType == NexFileVarType.WAVEFORM:
                self._ReadWaveforms(var)
            elif varType == NexFileVarType.POPULATION_VECTOR:
                self._ReadPopVectors(var)
            elif varType == NexFileVarType.CONTINUOUS:
                self._ReadContinuous(var)
            elif varType == NexFileVarType.MARKER:
                self._ReadMarker(var)

    def _ReadNex5FileHeader(self):
        fileHeaderFormat = '<i i 256s d q i Q q 56s'
        fileHeaderFormatSize = struct.calcsize(fileHeaderFormat)
        fhValues = struct.unpack(fileHeaderFormat, self.theFile.read(struct.calcsize(fileHeaderFormat)))
        keys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'NumVars', 'MetaOffset', 'End',
                'Padding']
        fileHeader = dict(zip(keys, fhValues))
        del fileHeader['Padding']

        if fileHeader['MagicNumber'] != 894977358:
            raise ValueError('Invalid .nex5 file')

        fileHeader['Comment'] = fileHeader['Comment'].decode('utf-8').strip('\x00')
        self.tsFreq = fileHeader['Frequency']
        self.fromTicksToSeconds = 1.0 / self.tsFreq
        fileHeader['Beg'] /= self.tsFreq
        fileHeader['End'] /= self.tsFreq
        return fileHeader

    def _ReadFileHeader(self):
        fileHeaderFormat = '<i i 256s d i i i 260s'
        fileHeaderFormatSize = struct.calcsize(fileHeaderFormat)
        fhValues = struct.unpack(fileHeaderFormat, self.theFile.read(struct.calcsize(fileHeaderFormat)))
        keys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'End', 'NumVars', 'Padding']
        fileHeader = dict(zip(keys, fhValues))
        del fileHeader['Padding']

        if fileHeader['MagicNumber'] != 827868494:
            raise ValueError('Invalid .nex file')

        fileHeader['Comment'] = fileHeader['Comment'].decode('utf-8').strip('\x00')
        self.tsFreq = fileHeader['Frequency']
        self.fromTicksToSeconds = 1.0 / self.tsFreq
        fileHeader['Beg'] /= self.tsFreq
        fileHeader['End'] /= self.tsFreq
        return fileHeader

    def _ReadVarHeader(self):
        varHeaderFormat = '<i i 64s i i i i i i d d d d i i i d d 52s'
        varHeaderSize = struct.calcsize(varHeaderFormat)
        vhValues = struct.unpack(varHeaderFormat, self.theFile.read(varHeaderSize))
        keys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'Wire', 'Unit', 'Gain', 'Filter', 'XPos', 'YPos',
                'SamplingRate', 'ADtoMV', 'NPointsWave', 'NMarkers', 'MarkerLength', 'MVOffset', 'PreThrTime', 'Padding']
        varHeader = dict(zip(keys, vhValues))
        del varHeader['Padding']

        varHeader['Name'] = varHeader['Name'].decode().strip('\x00').strip()
        # add fields that are in nex5 only
        varHeader['TsDataType'] = 0
        varHeader['ContDataType'] = 0
        varHeader['ContFragIndexType'] = 0
        varHeader['MarkerDataType'] = 0
        return varHeader

    def _ReadNex5VarHeader(self):
        varHeaderFormat = '<i i 64s Q Q i i d 32s d d Q d i i i i 60s'
        varHeaderSize = struct.calcsize(varHeaderFormat)
        vhValues = struct.unpack(varHeaderFormat, self.theFile.read(varHeaderSize))
        keys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'TsDataType', 'ContDataType', 'SamplingRate', 'Units',
                'ADtoMV', 'MVOffset', 'NPointsWave', 'PreThrTime', 'MarkerDataType', 'NMarkers', 'MarkerLength',
                'ContFragIndexType', 'Padding']
        varHeader = dict(zip(keys, vhValues))
        del varHeader['Padding']

        varHeader['Name'] = varHeader['Name'].decode().strip('\x00').strip()
        varHeader['Units'] = varHeader['Units'].decode().strip('\x00').strip()
        if varHeader['ContDataType'] == 1:
            varHeader['ADtoMV'] = 1
            varHeader['MVOffset'] = 0
        return varHeader

    def _ReadTimestamps(self, var):
        tsValueType = 'l'
        if var['Header']['TsDataType'] == 1:
            tsValueType = 'q'
        var['Timestamps'] = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)

    def _ReadAndScaleValuesUsingNumpy(self, valueType, count, coeff=1.0, divide=False):
        import numpy as np
        if valueType == 'h': numpyType = np.int16
        if valueType == 'l': numpyType = np.int32
        if valueType == 'L': numpyType = np.uint32
        if valueType == 'q': numpyType = np.int64
        if valueType == 'f': numpyType = np.float32
        if valueType == 'd': numpyType = np.float64
        values = np.fromfile(self.theFile, numpyType, count)
        if coeff == 1.0:
            return values
        if divide:
            return values / coeff
        else:
            return values * coeff
    
    def _ReadAndScaleValues(self, valueType, count, coeff=1.0, divide=False):
        if self.useNumpy:
            return self._ReadAndScaleValuesUsingNumpy(valueType, count, coeff, divide)
        if valueType == 'q':
            # 64-bit int is NOT supported by Python array, we have to use struct
            vList = []
            for i in range(count):
                vList.append(struct.unpack('q', self.theFile.read(8))[0])
            if coeff == 1.0:
                return vList
            if divide:
                return [x / coeff for x in vList]
            else:
                return [x * coeff for x in vList]
        else:
            values = array.array(valueType)
            values.fromfile(self.theFile, count)
        
        if coeff == 1.0:
            return values.tolist()
        
        if divide:
            return [x / coeff for x in values]
        else:
            return [x * coeff for x in values]

    def _ReadIntervals(self, var):
        if var['Header']['Count'] == 0:
            var['Intervals'] = [[], []]
            return
        tsValueType = 'l'
        if var['Header']['TsDataType'] == 1:
            tsValueType = 'q'
        intStarts = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)
        intEnds = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)
        var['Intervals'] = [intStarts, intEnds]

    def _ReadWaveforms(self, var):
        if var['Header']['NPointsWave'] <= 0:
            raise ValueError('invalid waveform header: NPointsWave is not positive')
        self._ReadTimestamps(var)
        wfValueType = 'h'
        coeff = var['Header']['ADtoMV']
        woffset = var['Header']['MVOffset']
        if var['Header']['ContDataType'] == 1:
            wfValueType = 'f'
            coeff = 1.0
            woffset = 0.0
        wf = self._ReadAndScaleValues(wfValueType, var['Header']['Count'] * var['Header']['NPointsWave'], coeff)
        if self.useNumpy:
            import numpy as np
            var['WaveformValues'] = wf.reshape( var['Header']['Count'], var['Header']['NPointsWave'])
            if woffset != 0:
                var['WaveformValues'] = var['WaveformValues'] + woffset
        else:
            if woffset != 0:
                var['WaveformValues'] = list(self._Chunks([x + woffset for x in wf], var['Header']['NPointsWave']))
            else:
                var['WaveformValues'] = list(self._Chunks(wf, var['Header']['NPointsWave']))

    def _Chunks(self, theList, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(theList), n):
            yield theList[i:i + n]

    def _ReadPopVectors(self, var):
        var['Weights'] = self._ReadAndScaleValues('d', var['Header']['Count'])

    def _ReadContinuous(self, var):
        tsValueType = 'l'
        if var['Header']['TsDataType'] == 1:
            tsValueType = 'q'
        var['FragmentTimestamps'] = self._ReadAndScaleValues(tsValueType, var['Header']['Count'], self.tsFreq, True)
        indexValueType = 'l'
        if var['Header']['ContFragIndexType'] == 1:
            indexValueType = 'q'
        var['FragmentIndexes'] = self._ReadAndScaleValues(indexValueType, var['Header']['Count'])
        var['FragmentCounts'] = []
        for frag in range(len(var['FragmentIndexes'])):
            if frag < var['Header']['Count'] - 1:
                count = var['FragmentIndexes'][frag+1] - var['FragmentIndexes'][frag]
            else:
                count = var['Header']['NPointsWave'] - var['FragmentIndexes'][frag]
            var['FragmentCounts'].append(count)
        contValueType = 'h'
        coeff = var['Header']['ADtoMV']
        woffset = var['Header']['MVOffset']
        if var['Header']['ContDataType'] == 1:
            contValueType = 'f'
            coeff = 1.0
            woffset = 0.0
        var['ContinuousValues'] = self._ReadAndScaleValues(contValueType, var['Header']['NPointsWave'], coeff)
        if woffset != 0:
            var['ContinuousValues'] = [x + woffset for x in var['ContinuousValues']]

    def _ReadMarker(self, var):
        self._ReadTimestamps(var)
        var['Fields'] = []
        var['MarkerFieldNames'] = []
        var['Markers'] = []
        isNumeric = True
        for field in range(var['Header']['NMarkers']):
            field = {'Name': self.theFile.read(64).decode().strip('\x00').strip()}
            var['MarkerFieldNames'].append(field['Name'])
            if var['Header']['MarkerDataType'] == 0:
                field['Markers'] = [self.theFile.read(var['Header']['MarkerLength']).decode().strip('\x00') for m in
                                   range(var['Header']['Count'])]
                if isNumeric:
                    for m in field['Markers']:
                        try:
                            n = int(m)
                        except ValueError:
                            isNumeric = False
                            break

                var['Fields'].append(field)
            else:
                field['Markers'] = self._ReadAndScaleValues('L', var['Header']['Count'])
                var['Fields'].append(field)
        # convert to numbers if all fields contain numbers to have the same values as in nex python interface
        if isNumeric:
            for f in var['Fields']:
                # we can use int (not uint that does not exist in Python anyway) since integers have no limit in Python
                f['Markers'] = [int(m) for m in f['Markers']]
        for f in var['Fields']:
            var['Markers'].append(f['Markers'])


class NexWriter(object):
    """
    Nex file writer class.
    Sample code:

    import nexfile
    w = nexfile.NexWriter(100000)
    w.fileData['FileHeader']['Comment'] = 'this is a comment'
    w.AddNeuron('neuron1', [1, 2, 3, 4])
    w.AddContVarWithSingleFragment('cont1', 2, 10000, [5, 6, 7, 8])
    w.WriteNexFile('C:\\Data\\testFileWrittenInPython.nex')
    w.WriteNex5File('C:\\Data\\testFileWrittenInPython.nex5', 1)

    """
    def __init__(self, timestampFrequency, useNumpy=False):
        """
        Constructor
        :param timestampFrequency: timestamp frequency in Hertz. Timestamps are stored as integers representing
                number of ticks, where tick = 1.0/timestampFrequency
        :param useNumpy: option to use numpy
        """
        self.theFile = None
        self.tsFreq = timestampFrequency
        self.useNumpy = useNumpy
        self.fromTicksToSeconds = 1.0/timestampFrequency

        self.varHeaderKeys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'Wire', 'Unit', 'Gain', 'Filter', 'XPos', 'YPos',
                'SamplingRate', 'ADtoMV', 'NPointsWave', 'NMarkers', 'MarkerLength', 'MVOffset', 'PreThrTime', 'Padding']

        self.nex5VarHeaderKeys = ['Type', 'Version', 'Name', 'DataOffset', 'Count', 'TsDataType', 'ContDataType',
                                  'SamplingRate', 'Units',
                'ADtoMV', 'MVOffset', 'NPointsWave', 'PreThrTime', 'MarkerDataType', 'NMarkers', 'MarkerLength',
                'ContFragIndexType', 'Padding']

        self.fileHeaderKeys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'End', 'NumVars', 'Padding']
        self.fileHeaderKeysToWrite = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'BegTicks', 'EndTicks', 'NumVars', 'Padding']
        self.nex5fileHeaderKeys = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'Beg', 'NumVars', 'MetaOffset', 'End',
                'Padding']
        self.nex5fileHeaderKeysToWrite = ['MagicNumber', 'NexFileVersion', 'Comment', 'Frequency', 'BegTicks', 'NumVars',
                                   'MetaOffset', 'EndTicks', 'Padding']
        # we will add nex5 file header keys (add MetaOffset)
        fhValues = [827868494, 106, '', timestampFrequency, 0, 0, 0, 0, '']
        fileHeader = dict(zip(self.nex5fileHeaderKeys, fhValues))
        self.fileData = {'FileHeader': fileHeader, 'Variables': []}

    def AddNeuron(self, name, timestamps, wire=0, unit=0, xpos=0, ypos=0):
        """
        Adds neuron file variable
        :param name: neuron name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :param wire: wire (electrode) number
        :param unit: unit number
        :param xpos: x position in [0, 100] range (used in 3d displays)
        :param ypos: y position in [0, 100] range (used in 3d displays)
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
        vhValues = [NexFileVarType.NEURON, 100, name, 0, len(timestamps), wire, unit, 0, 0, xpos, ypos, 0, 0, 0, 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps
        self.fileData['Variables'].append(var)

    def AddEvent(self, name, timestamps):
        """
        Adds event file variable
        :param name: event name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
        vhValues = [NexFileVarType.EVENT, 100, name, 0, len(timestamps), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps
        self.fileData['Variables'].append(var)

    def AddIntervalVariable(self, name, intStarts, intEnds):
        """
        Adds interval variable to file data
        :param name: variable name
        :param intStarts: list interval starts in seconds or numpy array if numpy option is specified in constructor
        :param intEnds: list interval ends in seconds or numpy array if numpy option is specified in constructor
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Interval starts', intStarts)
            self._VerifyIsNumpyArray('Interval ends', intEnds)
        vhValues = [NexFileVarType.INTERVAL, 100, name, 0, len(intStarts), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Intervals'] = [intStarts, intEnds]
        self.fileData['Variables'].append(var)

    def AddContVarWithSingleFragment(self, name, timestampOfFirstDataPoint, SamplingRate, values):
        """
        Adds continuous variable with a single fragment
        :param name: variable name
        :param timestampOfFirstDataPoint: time of first data point in seconds
        :param SamplingRate: sampling rate in Hz
        :param values: list of variable values in mV or numpy array of values if numpy option is specified in constructor
        :return: none
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Values', values)
        if SamplingRate <= 0 or SamplingRate > self.tsFreq:
            raise ValueError('invalid sampling rate in continuous')
        vhValues = [NexFileVarType.CONTINUOUS, 100, name, 0, 1, 0, 0, 0, 0, 0, 0, SamplingRate, 1.0,
                    len(values), 0, 0, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = [timestampOfFirstDataPoint]
        if self.useNumpy:
            import numpy as np
            var['Timestamps'] = np.array(var['Timestamps'])
        var['FragmentIndexes'] = [0]
        var['FragmentCounts'] = [len(values)]
        var['ContinuousValues'] = values
        self.fileData['Variables'].append(var)

    def AddContVarWithMultipleFragments(self, name, timestamps, SamplingRate, fragmentValues):
        """
        Adds continuous variable with multiple fragments
        :param name: variable name
        :param timestamps: list fragment start times in seconds or numpy array of fragment start times
                     if numpy option is specified in constructor
        :param SamplingRate: sampling rate in Hz
        :param fragmentValues: list of lists, each sublist is array data values in mV (or list of numpy arrays)
        :return:
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
            for fragment in fragmentValues:
                self._VerifyIsNumpyArray('FragmentValues', fragment)
        if SamplingRate <= 0 or SamplingRate > self.tsFreq:
            raise ValueError('invalid sampling rate in continuous')
        if len(timestamps) != len(fragmentValues):
            raise ValueError('should be the same number of timestamps and fragments')
        totalValues = 0
        var = {'FragmentIndexes': [], 'FragmentCounts': [], 'ContinuousValues': []}
        for fragment in fragmentValues:
            var['FragmentIndexes'].append(totalValues)
            var['FragmentCounts'].append(len(fragment))
            if not self.useNumpy:
                var['ContinuousValues'].extend(fragment)
            totalValues += len(fragment)
        if self.useNumpy:
            import numpy as np
            var['ContinuousValues'] = np.array([])
            for fragment in fragmentValues:
                var['ContinuousValues'] = np.append(var['ContinuousValues'], fragment)

        vhValues = [NexFileVarType.CONTINUOUS, 100, name, 0, 1, 0, 0, 0, 0, 0, 0, SamplingRate, 1.0,
                    totalValues, 0, 0, 0, 0, '']
        var['Header'] = dict(zip(self.varHeaderKeys, vhValues))
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps
        self.fileData['Variables'].append(var)

    def AddMarker(self, name, timestamps, fieldNames, markerFields):
        """
        Adds marker variable.
        :param name: variable name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :param fieldNames: list of field names
        :param markerFields: a list of lists (one list of numbers or strings per marker field)
        :return:
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
        if len(fieldNames) != len(markerFields):
            raise ValueError('different number of field names and field values')
        for x in markerFields:
            if len(x) != len(timestamps):
                raise ValueError('different number of timestamps and field values in a single field')
        vhValues = [NexFileVarType.MARKER, 100, name, 0, len(timestamps), 0, 0, 0, 0, 0, 0, 0, 0, 0, len(fieldNames), 6, 0, 0, '']
        var = {'Header': dict(zip(self.varHeaderKeys, vhValues))}
        self._AddNex5VarHeaderFields(var)
        var['Header']['MarkerDataType'] = 1
        var['Timestamps'] = timestamps
        var['MarkerFieldNames'] = fieldNames
        var['Markers'] = markerFields
        self._CalcMarkerLength(var)
        self.fileData['Variables'].append(var)

    def AddWave(self, name, timestamps, SamplingRate, WaveformValues, NPointsWave=0, PrethresholdTimeInSeconds=0, wire=0, unit=0):
        """
        Adds waveform variable.
        :param name: variable name
        :param timestamps: list of timestamps in seconds or numpy array if numpy option is specified in constructor
        :param SamplingRate: sampling rate of w/f values in Hz
        :param WaveformValues: a list of lists, each sublist contains values of a single waveform in mV;
               if numpy option is specified in constructor, numpy matrix
        :param NPointsWave: number of data points in each wave
        :param PrethresholdTimeInSeconds: pre-threshold time in seconds
        :param wire: wire (electrode) number
        :param unit: unit number
        :return:
        """
        if self.useNumpy:
            self._VerifyIsNumpyArray('Timestamps', timestamps)
            self._VerifyIsNumpyArray('WaveformValues', WaveformValues)
        if len(timestamps) != len(WaveformValues):
            raise ValueError('different number of timestamps and number of waveforms')
        if SamplingRate <= 0 or SamplingRate > self.tsFreq:
            raise ValueError('invalid sampling rate in wave')
        var = {}
        if len(WaveformValues) > 0 and len(WaveformValues[0]) > 0:
            NPointsWave = len(WaveformValues[0])
        if NPointsWave == 0:
            raise ValueError('invalid number of data points in wave')
        vhValues = [NexFileVarType.WAVEFORM, 100, name, 0, len(timestamps), wire, unit, 0, 0, 0, 0, SamplingRate, 1.0, NPointsWave, 0, 0, 0, PrethresholdTimeInSeconds, '']
        var['Header'] = dict(zip(self.varHeaderKeys, vhValues))
        self._AddNex5VarHeaderFields(var)
        var['Timestamps'] = timestamps
        var['WaveformValues'] = WaveformValues
        self.fileData['Variables'].append(var)

    def WriteNexFile(self, filePath):
        """
        Writes file data as .nex file.
        :param filePath: full path of file
        :return: none
        """
        self.theFile = open(filePath, 'wb')
        self.fileData['FileHeader']['MagicNumber'] = 827868494
        self.fileData['FileHeader']['NexFileVersion'] = 106
        numberOfVariables = len(self.fileData['Variables'])
        self.fileData['FileHeader']['NumVars'] = numberOfVariables

        maxTs = self._MaximumTimestamp()
        if round(maxTs * self.tsFreq) > pow(2, 31):
            raise ValueError('unable to save as .nex file: max timestamp exceeds 32-bit range; you can save as.nex5 file instead')
        self.fileData['FileHeader']['BegTicks'] = int(round(self.fileData['FileHeader']['Beg'] * self.tsFreq))
        self.fileData['FileHeader']['EndTicks'] = int(round(maxTs * self.tsFreq))

        for v in self.fileData['Variables']:
            v['Header']['TsDataType'] = 0
            v['Header']['Version'] = 102
            v['Header']['ContDataType'] = 0
            v['Header']['MarkerDataType'] = 0
            self._CalculateScaling(v)
            self._CalcMarkerLength(v)

        dataOffset = 544 + numberOfVariables*208
        for v in self.fileData['Variables']:
            v['Header']['Count'] = self._VarCount(v)
            v['Header']['DataOffset'] = dataOffset
            dataOffset += self._VarNumDataBytes(v)

        fileHeaderFormat = '<i <i 256s <d <i <i <i 260s'.split()
        for i in range(len(self.fileHeaderKeysToWrite)):
            self._WriteField(fileHeaderFormat[i], self.fileData['FileHeader'][self.fileHeaderKeysToWrite[i]])

        varHeaderFormat = '<i <i 64s <i <i <i <i <i <i <d <d <d <d <i <i <i <d <d 52s'.split()
        for v in self.fileData['Variables']:
            for i in range(len(self.varHeaderKeys)):
                self._WriteField(varHeaderFormat[i], v['Header'][self.varHeaderKeys[i]])

        for v in self.fileData['Variables']:
            self._VarWriteData(v)

        self.theFile.close()

    def WriteNex5File(self, filePath, saveContValuesAsFloats=0):
        """
        Writes file data as .nex5 file.
        :param filePath: full path of file
        :param saveContValuesAsFloats: if zero, continuous values are saved as 16-bit integers; if 1, saved as floats
        :return:
        """
        self.theFile = open(filePath, 'wb')
        self.fileData['FileHeader']['MagicNumber'] = 894977358
        self.fileData['FileHeader']['NexFileVersion'] = 501
        nvars = len(self.fileData['Variables'])
        self.fileData['FileHeader']['NumVars'] = nvars

        maxTs = self._MaximumTimestamp()
        tsAs64 = 0
        if round(maxTs * self.tsFreq) > pow(2, 31):
            tsAs64 = 1
            self.fileData['FileHeader']['NexFileVersion'] = 502

        for v in self.fileData['Variables']:
            v['Header']['TsDataType'] = tsAs64
            v['Header']['Version'] = 500
            v['Header']['ContDataType'] = saveContValuesAsFloats
            if v['Header']['Type'] == NexFileVarType.MARKER:
                self._CalcMarkerLength(v)
                if v['AllNumbers']:
                    v['Header']['MarkerDataType'] = 1
                else:
                    v['Header']['MarkerDataType'] = 0
            self._CalculateScaling(v)

        self.fileData['FileHeader']['BegTicks'] = int(round(self.fileData['FileHeader']['Beg'] * self.tsFreq))
        self.fileData['FileHeader']['EndTicks'] = int(round(maxTs * self.tsFreq))

        dataOffset = 356 + nvars * 244
        for v in self.fileData['Variables']:
            v['Header']['Count'] = self._VarCount(v)
            v['Header']['DataOffset'] = dataOffset
            dataOffset += self._VarNumDataBytes(v)

        fileHeaderFormat = '<i <i 256s <d <q <i <Q <q 56s'.split()
        for i in range(len(self.nex5fileHeaderKeysToWrite)):
            self._WriteField(fileHeaderFormat[i], self.fileData['FileHeader'][self.nex5fileHeaderKeysToWrite[i]])

        varHeaderFormat = '<i <i 64s <Q <Q <i <i <d 32s <d <d <Q <d <i <i <i <i 60s'.split()
        for v in self.fileData['Variables']:
            for i in range(len(self.nex5VarHeaderKeys)):
                self._WriteField(varHeaderFormat[i], v['Header'][self.nex5VarHeaderKeys[i]])

        for v in self.fileData['Variables']:
            self._VarWriteData(v)

        metaData = {"file": {}, 'variables': []}
        metaData["file"]["writerSoftware"] = {}
        metaData["file"]["writerSoftware"]["name"] = 'nexfile.py'
        metaData["file"]["writerSoftware"]["version"] = 'May-06-2017'
        for v in self.fileData['Variables']:
            varMeta = {'name': v['Header']['Name']}
            if v['Header']['Type'] == NexFileVarType.NEURON or v['Header']['Type'] == NexFileVarType.WAVEFORM:
                varMeta['unitNumber'] = v['Header']['Unit']
                varMeta['probe'] = {}
                varMeta['probe']['wireNumber'] = v['Header']['Wire']
                varMeta['probe']['position'] = {}
                varMeta['probe']['position']['x'] = v['Header']['XPos']
                varMeta['probe']['position']['y'] = v['Header']['YPos']
            metaData['variables'].append(varMeta)

        metaString = json.dumps(metaData).encode('utf-8')
        pos = self.theFile.tell()
        self.theFile.write(metaString)
        metaPosInHeader = 284
        self.theFile.seek(metaPosInHeader, 0)
        self.theFile.write(struct.pack('<Q', pos))

        self.theFile.close()

    # the following class methods are internal
    def _VerifyIsNumpyArray(self, name, a):
        import numpy as np
        if not isinstance(a, np.ndarray):
            raise ValueError(name + ' should be a numpy array')

    def _ConvertStringToBytesIfNeeded(self, stringOrBytes):
        if isinstance(stringOrBytes, bytes):
            return stringOrBytes
        else:
            return stringOrBytes.encode('utf-8')

    def _WriteField(self, theFormat, theField):
        if theFormat.endswith('s'):
            theField = self._ConvertStringToBytesIfNeeded(theField)
        self.theFile.write(struct.pack(theFormat, theField))

    def _AddNex5VarHeaderFields(self, var):
        """
        Adds .nex5 variable header fields.
        :param var: file data variable
        :return: none
        """
        var['Header']['TsDataType'] = 0
        var['Header']['ContDataType'] = 0
        var['Header']['Units'] = ''
        var['Header']['MarkerDataType'] = 0
        var['Header']['ContFragIndexType'] = 0

    def _BytesInTimestamp(self, var):
        """
        Calculates number of bytes in timestamp.
        :param var: file data variable
        :return: number of bytes in timestamp
        """
        if var['Header']['TsDataType'] == 0:
            return 4
        else:
            return 8

    def _BytesInContValue(self, var):
        """
        :param var:  file data variable
        :return: number of bytes in continuous value
        """
        if var['Header']['ContDataType'] == 0:
            return 2
        else:
            return 4

    def _VarNumDataBytes(self, var):
        """
        :param var: file data variable
        :return: number of bytes in variable data
        """
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
            return self._BytesInTimestamp(var) * len(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return 2 * self._BytesInTimestamp(var) * len(var['Intervals'][0])
        elif varType == NexFileVarType.WAVEFORM:
            return self._BytesInTimestamp(var) * len(var['Timestamps']) + self._BytesInContValue(var) * len(var['Timestamps']) * var['Header']['NPointsWave']
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            return (self._BytesInTimestamp(var) + 4) * len(var['Timestamps']) + self._BytesInContValue(var) * len(var['ContinuousValues'])
        elif varType == NexFileVarType.MARKER:
            if var['Header']['MarkerDataType'] == 0:
                singleMarkerLength = var['Header']['MarkerLength']
            else:
                singleMarkerLength = 4
            nts = len(var['Timestamps'])
            nm = len(var['MarkerFieldNames'])
            return self._BytesInTimestamp(var) * nts + nm * (64 + nts * singleMarkerLength)

    def _VarCount(self, var):
        """
        Calculates count field for file variables
        :param var:
        :return:
        """
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
            return len(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return len(var['Intervals'][0])
        elif varType == NexFileVarType.WAVEFORM:
            return len(var['Timestamps'])
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            return len(var['Timestamps'])
        elif varType == NexFileVarType.MARKER:
            return len(var['Timestamps'])

    def _MaxOfNumpyArrayOrZero(self, x):
        import numpy as np
        if len(x) == 0:
            return 0
        else:
            return np.max(x)

    def _VarMaxTimestampNumpy(self, var):
        import numpy
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT or varType == NexFileVarType.MARKER:
            return self._MaxOfNumpyArrayOrZero(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return self._MaxOfNumpyArrayOrZero(var['Intervals'][1])
        elif varType == NexFileVarType.WAVEFORM:
            if len(var['Timestamps']) == 0:
                return 0
            return self._MaxOfNumpyArrayOrZero(var['Timestamps']) + (var['Header']['NPointsWave']-1)/var['Header']['SamplingRate']
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            if len(var['Timestamps']) == 0:
                return 0
            return var['Timestamps'][-1] + (var['FragmentCounts'][-1] - 1) / var['Header']['SamplingRate']

    def _MaxValueOrZero(self, theList):
        if len(theList) > 0:
            return max(theList)
        return 0

    def _VarMaxTimestamp(self, var):
        if self.useNumpy:
            return self._VarMaxTimestampNumpy(var)
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT or varType == NexFileVarType.MARKER:
            return self._MaxValueOrZero(var['Timestamps'])
        elif varType == NexFileVarType.INTERVAL:
            return self._MaxValueOrZero(var['Intervals'][1])
        elif varType == NexFileVarType.WAVEFORM:
            if len(var['Timestamps']) == 0:
                return 0
            return max(var['Timestamps']) + (var['Header']['NPointsWave']-1)/var['Header']['SamplingRate']
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return 0
        elif varType == NexFileVarType.CONTINUOUS:
            if len(var['Timestamps']) == 0:
                return 0
            return var['Timestamps'][-1] + (var['FragmentCounts'][-1] - 1) / var['Header']['SamplingRate']

    def _VarWriteTimestampsNumpy(self, var, timestamps):
        import numpy as np
        if self._BytesInTimestamp(var) == 4:
            np.round(timestamps * self.tsFreq).astype(np.int32).tofile(self.theFile)
        else:
            np.round(timestamps * self.tsFreq).astype(np.int64).tofile(self.theFile)

    def _VarWriteTimestamps(self, var, timestamps):
        if self.useNumpy:
            return self._VarWriteTimestampsNumpy(var, timestamps)
        if self._BytesInTimestamp(var) == 4:
            tsTicks = [int(round(x * self.tsFreq)) for x in timestamps]
            values = array.array('l', tsTicks)
            values.tofile(self.theFile)
        else:
            for x in timestamps:
                if sys.version_info < (3,):
                    self.theFile.write(struct.pack('q', long(round(x * self.tsFreq))))
                else:
                    self.theFile.write(struct.pack('q', int(round(x * self.tsFreq))))

    def _VarWriteWaveformsNumpy(self, var):
        import numpy as np
        if self._BytesInContValue(var) == 2:
            np.round(var['WaveformValues'] / var['Header']['ADtoMV']).astype(np.int16).tofile(self.theFile)
        else:
            var['WaveformValues'].astype(np.float32).tofile(self.theFile)

    def _VarWriteContinuousValuesNumpy(self, var):
        import numpy as np
        if self._BytesInContValue(var) == 2:
            np.round(var['ContinuousValues'] / var['Header']['ADtoMV']).astype(np.int16).tofile(self.theFile)
        else:
            var['ContinuousValues'].astype(np.float32).tofile(self.theFile)

    def _VarWriteData(self, var):
        varType = var['Header']['Type']
        if varType == NexFileVarType.NEURON or varType == NexFileVarType.EVENT:
            self._VarWriteTimestamps(var, var['Timestamps'])
            return
        elif varType == NexFileVarType.INTERVAL:
            for i in range(2):
                self._VarWriteTimestamps(var, var['Intervals'][i])
            return
        elif varType == NexFileVarType.WAVEFORM:
            self._VarWriteTimestamps(var, var['Timestamps'])
            if self.useNumpy:
                self._VarWriteWaveformsNumpy(var)
                return
            if self._BytesInContValue(var) == 2:
                for w in var['WaveformValues']:
                    waveValues = [int(x / var['Header']['ADtoMV']) for x in w]
                    values = array.array('h', waveValues)
                    values.tofile(self.theFile)
            else:
                for w in var['WaveformValues']:
                    values = array.array('f', w)
                    values.tofile(self.theFile)
            return
        elif varType == NexFileVarType.POPULATION_VECTOR:
            return
        elif varType == NexFileVarType.CONTINUOUS:
            self._VarWriteTimestamps(var, var['Timestamps'])
            values = array.array('l', var['FragmentIndexes'])
            values.tofile(self.theFile)
            if self.useNumpy:
                self._VarWriteContinuousValuesNumpy(var)
                return
            if self._BytesInContValue(var) == 2:
                contValues = [int(x / var['Header']['ADtoMV']) for x in var['ContinuousValues']]
                values = array.array('h', contValues)
                values.tofile(self.theFile)
            else:
                values = array.array('f', var['ContinuousValues'])
                values.tofile(self.theFile)
            return
        elif varType == NexFileVarType.MARKER:
            self._VarWriteTimestamps(var, var['Timestamps'])
            for i, name in enumerate(var['MarkerFieldNames']):
                self._WriteField('64s', name)
                if var['Header']['MarkerDataType'] == 0:
                    for v in var['Markers'][i]:
                        if isinstance(v, numbers.Number):
                            sv = '{0:05d}'.format(v) + '\x00'
                        else:
                            sv = v
                            while len(sv) < var['Header']['MarkerLength']:
                                sv += '\x00'
                        self.theFile.write(sv.encode('utf-8'))
                else:
                    values = array.array('L', var['Markers'][i])
                    values.tofile(self.theFile)
            return

    def _CalcMarkerLength(self, var):
        if var['Header']['Type'] != NexFileVarType.MARKER:
            return
        maxStringLength = 0
        allNumbers = True
        for field in var['Markers']:
            for x in field:
                if not isinstance(x, numbers.Number):
                    allNumbers = False
                    if not isinstance(x, str):
                        raise ValueError('marker values should be either numbers or strings')
                    maxStringLength = max(maxStringLength, len(x))
        var['AllNumbers'] = allNumbers
        if allNumbers:
            var['Header']['MarkerLength'] = 6
        else:
            var['Header']['MarkerLength'] = max(6, maxStringLength + 1)

    def _MaximumTimestamp(self):
        maxTs = 0
        for v in self.fileData['Variables']:
            maxTs = max(maxTs, self._VarMaxTimestamp(v))
        return maxTs

    def _SignalAbsMaxNumPy(self, signal):
        import numpy as np
        return np.max(np.abs(signal))

    def _CalculateScaling(self, var):
        varType = var['Header']['Type']
        if varType == NexFileVarType.CONTINUOUS:
            if var['Header']['ContDataType'] == 1:
                var['Header']['ADtoMV'] = 1
                return
            contMax = 0
            if self.useNumpy:
                contMax = self._SignalAbsMaxNumPy(var['ContinuousValues'])
            else:
                contMax = max(contMax, abs(max(var['ContinuousValues'])))
                contMax = max(contMax, abs(min(var['ContinuousValues'])))
            if contMax == 0:
                var['Header']['ADtoMV'] = 1
            else:
                var['Header']['ADtoMV'] = contMax / 32767.0
            return
        if varType == NexFileVarType.WAVEFORM:
            if var['Header']['ContDataType'] == 1:
                var['Header']['ADtoMV'] = 1
                return
            waveMax = 0
            if self.useNumpy:
                waveMax = self._SignalAbsMaxNumPy(var['WaveformValues'])
            else:
                for w in var['WaveformValues']:
                    waveMax = max(waveMax, abs(max(w)))
                    waveMax = max(waveMax, abs(min(w)))
            if waveMax == 0:
                var['Header']['ADtoMV'] = 1
            else:
                var['Header']['ADtoMV'] = waveMax / 32767.0
