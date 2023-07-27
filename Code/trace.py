#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML PolyData Reader'
data_alldatNDnet_s352upNDsklS010vtp = XMLPolyDataReader(FileName=['/home/ankit/Python_Environments/EAGLE/SCRIPTS/Data_all.dat.NDnet_s3.52.up.NDskl.S010.vtp'])
data_alldatNDnet_s352upNDsklS010vtp.CellArrayStatus = ['arc_id', 'type', 'down_index', 'up_index', 'length', 'flags']
data_alldatNDnet_s352upNDsklS010vtp.PointArrayStatus = ['persistence_ratio', 'persistence_nsigmas', 'persistence', 'persistence_pair', 'parent_index', 'parent_log_index', 'log_field_value', 'field_value', 'cell', 'critical_index']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1305, 576]

# show data in view
data_alldatNDnet_s352upNDsklS010vtpDisplay = Show(data_alldatNDnet_s352upNDsklS010vtp, renderView1)
# trace defaults for the display properties.
data_alldatNDnet_s352upNDsklS010vtpDisplay.ColorArrayName = [None, '']

# reset view to fit data
renderView1.ResetCamera()

# set scalar coloring
ColorBy(data_alldatNDnet_s352upNDsklS010vtpDisplay, ('CELLS', 'length'))

# rescale color and/or opacity maps used to include current data range
data_alldatNDnet_s352upNDsklS010vtpDisplay.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
data_alldatNDnet_s352upNDsklS010vtpDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'length'
lengthLUT = GetColorTransferFunction('length')
lengthLUT.RGBPoints = [-1.0, 0.231373, 0.298039, 0.752941, 20.41041950104118, 0.865003, 0.865003, 0.865003, 41.82083900208236, 0.705882, 0.0156863, 0.14902]
lengthLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'length'
lengthPWF = GetOpacityTransferFunction('length')
lengthPWF.Points = [-1.0, 0.0, 0.5, 0.0, 41.82083900208236, 1.0, 0.5, 0.0]
lengthPWF.ScalarRangeInitialized = 1

# create a new 'Threshold'
threshold1 = Threshold(Input=data_alldatNDnet_s352upNDsklS010vtp)
threshold1.Scalars = ['POINTS', 'cell']
threshold1.ThresholdRange = [28.0, 2899706.2]

# Properties modified on threshold1
threshold1.Scalars = ['CELLS', 'length']

# show data in view
threshold1Display = Show(threshold1, renderView1)
# trace defaults for the display properties.
threshold1Display.ColorArrayName = ['CELLS', 'length']
threshold1Display.LookupTable = lengthLUT
threshold1Display.ScalarOpacityUnitDistance = 16.949193609843075

# hide data in view
Hide(data_alldatNDnet_s352upNDsklS010vtp, renderView1)

# show color bar/color legend
threshold1Display.SetScalarBarVisibility(renderView1, True)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [4.4586258405146015, 69.24939910952811, 128.309368095823]
renderView1.CameraFocalPoint = [24.991113972701026, 25.00192019157112, 25.00201433361509]
renderView1.CameraViewUp = [0.26907112165069913, 0.9035319721388019, -0.33351267864445894]
renderView1.CameraParallelScale = 43.29150712998766

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).